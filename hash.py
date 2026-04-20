from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any


def sha256_file(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(file_path: Path) -> str:
    """Calcula el hash MD5 de un archivo."""
    h = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_basic_info(file_path: Path) -> dict[str, Any]:
    """Obtiene información básica del archivo."""
    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return {
        "nombre": file_path.name,
        "ruta": str(file_path.resolve()),
        "tamano_bytes": stat.st_size,
        "extension": file_path.suffix.lower(),
        "mime_type": mime_type or "desconocido",
    }


def detect_c2pa_signature(file_path: Path) -> bool:
    """
    Detección simple y preliminar de cadenas asociadas a C2PA.
    No sustituye una validación formal del manifiesto.
    """
    try:
        content = file_path.read_bytes()
        markers = [b"c2pa", b"application/c2pa", b"Content Credentials"]
        return any(marker.lower() in content.lower() for marker in markers)
    except Exception:
        return False


def classify_risk(file_info: dict[str, Any], has_c2pa: bool) -> str:
    """
    Clasificación orientativa de riesgo forense.
    """
    ext = file_info["extension"]

    if has_c2pa:
        return "bajo-moderado (existe indicio de provenance, requiere validacion)"
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".wav", ".mp3"]:
        return "moderado-alto (archivo multimedia sin provenance verificada)"
    return "indeterminado"


def analyze_file(file_path: Path) -> dict[str, Any]:
    """Ejecuta un análisis básico de triage forense."""
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    info = file_basic_info(file_path)
    sha256 = sha256_file(file_path)
    md5 = md5_file(file_path)
    has_c2pa = detect_c2pa_signature(file_path)

    report = {
        "archivo": info,
        "integridad": {
            "sha256": sha256,
            "md5": md5,
        },
        "provenance": {
            "indicio_c2pa": has_c2pa,
            "nota": (
                "La deteccion positiva solo indica posible presencia de metadatos o referencias C2PA. "
                "Debe validarse con herramientas especializadas."
            ),
        },
        "evaluacion_preliminar": {
            "nivel_riesgo": classify_risk(info, has_c2pa),
            "observacion": (
                "Este resultado es orientativo y no sustituye un peritaje forense completo."
            ),
        },
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage forense basico para archivos multimedia"
    )
    parser.add_argument("archivo", type=str, help="Ruta del archivo a analizar")
    parser.add_argument(
        "--salida",
        type=str,
        default="reporte_forense.json",
        help="Ruta del archivo JSON de salida",
    )
    args = parser.parse_args()

    file_path = Path(args.archivo)
    output_path = Path(args.salida)

    report = analyze_file(file_path)

    output_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Analisis completado.")
    print(f"Reporte guardado en: {output_path}")


if __name__ == "__main__":
    main()