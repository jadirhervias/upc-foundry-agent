# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_foundry_hosting import ResponsesHostServer
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).with_name(".env"), override=True)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)
load_dotenv(override=True)


def _get_server_status(executable: str) -> dict:
    """Return parsed JSON from ``foundry server status --output json``."""
    raw = subprocess.check_output(
        [executable, "server", "status", "--output", "json"], encoding="utf-8"
    )
    return json.loads(raw)


def _server_is_reachable(endpoint: str) -> bool:
    """Quick HTTP probe to see if the Foundry web server is responding."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{endpoint.rstrip('/')}/v1/models", timeout=5)
        return True
    except Exception:
        return False


def create_local_client(model: str) -> OpenAIChatCompletionClient:
    executable = shutil.which("foundry")
    if not executable:
        raise RuntimeError("Foundry Local CLI is not installed or not on PATH.")

    # --- ensure the server is running ----------------------------------------
    status = _get_server_status(executable)
    urls = status.get("webUrls")
    endpoint = urls[0] if isinstance(urls, list) and urls else None

    if not (endpoint and _server_is_reachable(endpoint)):
        # Try to start; tolerate failure if the server comes up anyway.
        result = subprocess.run([executable, "server", "start"])
        if result.returncode != 0:
            status = _get_server_status(executable)
            urls = status.get("webUrls")
            endpoint = urls[0] if isinstance(urls, list) and urls else None
            if not (endpoint and _server_is_reachable(endpoint)):
                raise RuntimeError(
                    f"Foundry Local server failed to start (exit {result.returncode}). "
                    "Run 'foundry server logs' for details."
                )

    # Refresh status after potential start
    status = _get_server_status(executable)
    urls = status.get("webUrls")
    if not isinstance(urls, list) or not urls:
        raise RuntimeError("Foundry Local did not report a running HTTP server.")
    endpoint = urls[0]
    if not isinstance(endpoint, str):
        raise RuntimeError("Foundry Local reported an invalid endpoint.")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise RuntimeError("Foundry Local endpoint must be a loopback HTTP address.")

    # --- ensure the model is loaded ------------------------------------------
    subprocess.run([executable, "model", "load", model], check=True)

    return OpenAIChatCompletionClient(
        model=model,
        async_client=AsyncOpenAI(
            base_url=f"{endpoint.rstrip('/')}/v1",
            api_key="foundry-local",
        ),
    )


def main():
    model_name = os.getenv("FOUNDRY_LOCAL_MODEL", "").strip() or "phi-4-mini-instruct-openvino-npu:4"
    client = create_local_client(model=model_name)

    asyncio.run(serve_agent(client))


async def serve_agent(client: OpenAIChatCompletionClient) -> None:
    aria_instructions = (
        "Eres ARIA, el Asistente de Resolución Inteligente y Automatizada para la Mesa de Ayuda TI (Nivel 1) "
        "de una entidad financiera.\n\n"
        "--- OBJETIVO OPERATIVO ---\n"
        "Tu función es realizar el triage, clasificación y resolución guiada de incidencias técnicas comunes "
        "(conectividad, ofimática, sistema operativo y herramientas de usuario final).\n\n"
        "--- FORMATO OBLIGATORIO DE RESPUESTA ---\n"
        "Estructura SIEMPRE tus respuestas exactamente con estas secciones:\n"
        "• **Categoría:** <Redes / Ofimática / Sistema Operativo>\n"
        "• **Prioridad:** <Baja / Media / Alta>\n"
        "• **Diagnóstico y Solución:** <Pasos claros y detallados para el colaborador>\n"
        "• **Acción:** <Resuelto en N1 o Requiere escalamiento a analista Nivel 2>\n\n"
        "--- PRINCIPIOS ÉTICOS Y DE SEGURIDAD (LEY 29733 Y LEY 31814) ---\n"
        "- Transparencia y no invención: Sé honesto y preciso. No inventes procedimientos ni sugieras comandos riesgosos.\n"
        "- Confidencialidad y anonimización: Queda terminantemente prohibido solicitar, procesar o mostrar contraseñas, tokens, credenciales o datos financieros confidenciales.\n"
        "- Responsabilidad y equidad: No reemplaces la supervisión humana. Si el problema afecta a sistemas transaccionales críticos o requiere permisos de administrador del dominio, debe indicarse el escalamiento a un analista de Nivel 2.\n"
        "- Registro auditable: Mantén siempre el formato indicado para garantizar la trazabilidad del 100% de las consultas."
    )

    async with Agent(
        client=client,
        instructions=aria_instructions,
        # History will be managed by the hosting infrastructure, thus there
        # is no need to store history by the service. Learn more at:
        # https://developers.openai.com/api/reference/resources/responses/methods/create
        default_options={"store": False},
    ) as agent:
        server = ResponsesHostServer(agent)
        await server.run_async()


if __name__ == "__main__":
    main()
