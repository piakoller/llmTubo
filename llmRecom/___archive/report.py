# @title Saving Report (DOCX/PDF) Helper Functions

import os
import subprocess


def convert_markdown(markdown_text, output_path, filename, file_type):
    """
    Converts markdown text to DOCX or PDF using pandoc.

    Args:
        markdown_text: The markdown text to convert.
        output_path: The directory where the output file should be saved.
        filename: The name of the output file (without extension).
        file_type: The desired output file type ('docx' or 'pdf').

    Raises:
        ValueError: If an invalid file type is specified.
        FileNotFoundError: If pandoc is not found in the system's PATH.
        subprocess.CalledProcessError: If the pandoc command fails.
        OSError: If there is an error during file operations.
    """
    os.makedirs(output_path, exist_ok=True)

    if file_type not in ["docx", "pdf"]:
        raise ValueError("Invalid file type specified. Must be 'docx' or 'pdf'.")

    docx_filepath = os.path.join(output_path, f"{filename}.docx")

    try:
        # Check if pandoc is available
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)

        # Convert Markdown to DOCX
        subprocess.run(
            ["pandoc", "-f", "markdown", "-t", "docx", "-o", docx_filepath],
            input=markdown_text,
            encoding="utf-8",
            check=True,
        )
        # print(f"DOCX file saved to: {docx_filepath}")

        if file_type == "pdf":
            pdf_filepath = os.path.join(output_path, f"{filename}.pdf")
            # Convert DOCX to PDF (using libreoffice on Colab)
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    output_path,
                    docx_filepath,
                ],
                check=True,
            )
            print(f"PDF file saved to: {pdf_filepath}")

            # Delete the temporary DOCX file
            os.remove(docx_filepath)
            print(f"Temporary DOCX file deleted: {docx_filepath}")

    except FileNotFoundError:
        raise FileNotFoundError(
            "pandoc not found. Please ensure it is installed and in your system's PATH."
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd, output=e.output, stderr=e.stderr
        )
    except OSError as e:
        raise OSError(f"Error during file operations: {e}")