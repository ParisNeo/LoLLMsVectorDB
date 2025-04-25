import hashlib
import re
import os
from pathlib import Path
from typing import List, Tuple
from lollmsvectordb.database_elements.document import Document

import pipmaster as pm

if not pm.is_installed("docling"):
    pm.install("docling")

class TextDocumentsLoader:
    def load_document(self, file_path: Path) -> Tuple[str, Document]:
        """Load a document, create and return a Document object."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        file_hash = hasher.hexdigest()
        
        title = os.path.basename(file_path)
        doc = Document(hash=file_hash, title=title, path=file_path)
        
        text = self.read_file(file_path)
        
        return text, doc

    @staticmethod
    def read_file(file_path: Path | str) -> str:
        """
        Read a file and return its content as a string.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The content of the file.

        Raises:
            ValueError: If the file type is unknown.
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            return TextDocumentsLoader.read_pdf_file(file_path)
        elif file_path.suffix.lower() == ".docx":
            return TextDocumentsLoader.read_docx_file(file_path)
        elif file_path.suffix.lower() == ".xlsx":
            return TextDocumentsLoader.read_xlsx_file(file_path)
        elif file_path.suffix.lower() == ".json":
            return TextDocumentsLoader.read_json_file(file_path)
        elif file_path.suffix.lower() == ".html":
            return TextDocumentsLoader.read_html_file(file_path)
        elif file_path.suffix.lower() == ".png":
            return TextDocumentsLoader.read_png_file(file_path)        
        elif file_path.suffix.lower() == ".bmp":
            return TextDocumentsLoader.read_bmp_file(file_path)        
        elif file_path.suffix.lower() == ".jpg" or file_path.suffix.lower() == ".jpeg":
            return TextDocumentsLoader.read_jpg_file(file_path)        
        elif file_path.suffix.lower() == ".pptx":
            return TextDocumentsLoader.read_pptx_file(file_path)
        elif file_path.suffix.lower() in [".pcap"]:
            return TextDocumentsLoader.read_pcap_file(file_path)
        elif file_path.suffix.lower() in [
            ".sh",
            ".json",
            ".sym",
            ".log",
            ".snippet",
            ".se",
            ".yml",
            ".snippets",
            ".lua",
            ".pdf",
            ".md",
            ".yaml",
            ".inc",
            ".txt",
            ".ini",
            ".pas",
            ".map",
            ".php",
            ".rtf",
            ".hpp",
            ".h",
            ".asm",
            ".xml",
            ".hh",
            ".sql",
            ".java",
            ".c",
            ".html",
            ".inf",
            ".rb",
            ".py",
            ".cs",
            ".js",
            ".bat",
            ".css",
            ".s",
            ".cpp",
            ".csv",
            ".vue",
            ".png",
            ".jpg",
            ".tiff",
            ".bmp"
        ]:
            return TextDocumentsLoader.read_text_file(file_path)
        elif file_path.suffix.lower() in [".msg"]:
            return TextDocumentsLoader.read_msg_file(file_path)
        else:
            raise ValueError("Unknown file type")

    @staticmethod
    def get_supported_file_types() -> List[str]:
        """
        Get the list of supported file types.

        Returns:
            List[str]: The list of supported file types.
        """
        return [
            ".sh",
            ".json",
            ".sym",
            ".log",
            ".snippet",
            ".se",
            ".yml",
            ".snippets",
            ".lua",
            ".pdf",
            ".md",
            ".docx",
            ".xlsx",
            ".png",
            ".bmp",
            ".jpg",
            ".yaml",
            ".inc",
            ".txt",
            ".ini",
            ".pas",
            ".pptx",
            ".map",
            ".php",
            ".xlsx",
            ".rtf",
            ".hpp",
            ".h",
            ".asm",
            ".xml",
            ".hh",
            ".sql",
            ".java",
            ".c",
            ".html",
            ".inf",
            ".rb",
            ".py",
            ".cs",
            ".js",
            ".bat",
            ".css",
            ".s",
            ".cpp",
            ".csv",
            ".msg",
        ]

    @staticmethod
    def read_pcap_file(file_path):
        import dpkt

        result = ""  # Create an empty string to store the packet details
        with open(file_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                eth = dpkt.ethernet.Ethernet(buf)

                # Extract Ethernet information
                src_mac = ":".join("{:02x}".format(b) for b in eth.src)
                dst_mac = ":".join("{:02x}".format(b) for b in eth.dst)
                eth_type = eth.type

                # Concatenate Ethernet information to the result string
                result += f"Timestamp: {timestamp}\n"
                result += f"Source MAC: {src_mac}\n"
                result += f"Destination MAC: {dst_mac}\n"
                result += f"Ethernet Type: {eth_type}\n"

                # Check if packet is an IP packet
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data

                    # Extract IP information
                    src_ip = dpkt.ip.inet_to_str(ip.src)
                    dst_ip = dpkt.ip.inet_to_str(ip.dst)
                    ip_proto = ip.p

                    # Concatenate IP information to the result string
                    result += f"Source IP: {src_ip}\n"
                    result += f"Destination IP: {dst_ip}\n"
                    result += f"IP Protocol: {ip_proto}\n"

                    # Check if packet is a TCP packet
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data

                        # Extract TCP information
                        src_port = tcp.sport
                        dst_port = tcp.dport

                        # Concatenate TCP information to the result string
                        result += f"Source Port: {src_port}\n"
                        result += f"Destination Port: {dst_port}\n"

                        # Add more code here to extract and concatenate other TCP details if needed

                    # Check if packet is a UDP packet
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data

                        # Extract UDP information
                        src_port = udp.sport
                        dst_port = udp.dport

                        # Concatenate UDP information to the result string
                        result += f"Source Port: {src_port}\n"
                        result += f"Destination Port: {dst_port}\n"

                        # Add more code here to extract and concatenate other UDP details if needed

                    # Add more code here to handle other protocols if needed

                result += "-" * 50 + "\n"  # Separator between packets

        return result  # Return the result string

    @staticmethod
    def read_pdf_file(file_path: Path) -> str:
        """
        Read a PDF file and return its content as Markdown.

        Args:
            file_path (Path): The path to the PDF file.

        Returns:
            str: The content of the PDF file in Markdown format.
        """
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text


    @staticmethod
    def read_docx_file(file_path: Path) -> str:
        """
        Read a DOCX file and return its content as a string.

        Args:
            file_path (Path): The path to the DOCX file.

        Returns:
            str: The content of the DOCX file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text


    @staticmethod
    def read_xlsx_file(file_path: Path) -> str:
        """
        Read a XLSX file and return its content as a string.

        Args:
            file_path (Path): The path to the XLSX file.

        Returns:
            str: The content of the XLSX file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text

    @staticmethod
    def read_json_file(file_path: Path) -> str:
        """
        Read a JSON file and return its content as a string.

        Args:
            file_path (Path): The path to the JSON file.

        Returns:
            str: The content of the JSON file.
        """
        import json

        with open(file_path, "r", encoding="utf-8") as file:
            data = str(json.load(file))
        return data

    @staticmethod
    def read_csv_file(file_path: Path) -> str:
        """
        Read a CSV file and return its content as a string.
        Args:
            file_path (Path): The path to the CSV file.
        Returns:
            str: The content of the CSV file.
        """
        with open(file_path, "r") as file:
            content = file.read()
        return content

    @staticmethod
    def read_html_file(file_path: Path) -> str:
        """
        Read an HTML file and return its content as a string.

        Args:
            file_path (Path): The path to the HTML file.

        Returns:
            str: The content of the HTML file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text

    @staticmethod
    def read_png_file(file_path: Path) -> str:
        """
        Read an HTML file and return its content as a string.

        Args:
            file_path (Path): The path to the png file.

        Returns:
            str: The content of the HTML file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text
    
    @staticmethod
    def read_bmp_file(file_path: Path) -> str:
        """
        Read an bmp file and return its content as a string.

        Args:
            file_path (Path): The path to the bmp file.

        Returns:
            str: The content of the bmp file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text
    
    @staticmethod
    def read_jpg_file(file_path: Path) -> str:
        """
        Read an jpg file and return its content as a string.

        Args:
            file_path (Path): The path to the jpg file.

        Returns:
            str: The content of the jpg file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text
    
    @staticmethod
    def read_pptx_file(file_path: Path) -> str:
        """
        Read a PPTX file and return its content as a string.

        Args:
            file_path (Path): The path to the PPTX file.

        Returns:
            str: The content of the PPTX file.
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return text

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """
        Read a text file and return its content as a string.

        Args:
            file_path (Path): The path to the text file.

        Returns:
            str: The content of the text file.
        """
        # Implementation details omitted for brevity
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        return content

    @staticmethod
    def read_msg_file(file_path: Path) -> str:
        """
        Read a text file and return its content as a string.

        Args:
            file_path (Path): The path to the text file.

        Returns:
            str: The content of the text file.
        """
        try:
            import extract_msg
        except ImportError:
            pm.install("extract-msg", force_reinstall=True, upgrade=True)
            import extract_msg
        # Implementation details omitted for brevity
        msg = extract_msg.Message(file_path)
        msg_message = msg.body
        return msg_message

    def load_msg_as_text(file_path):
        try:
            import extract_msg
        except ImportError:
            pm.install("extract-msg", force_reinstall=True, upgrade=True)
            import extract_msg
        msg = extract_msg.Message(file_path)
        # Extract the relevant parts of the email
        subject = f"Subject: {msg.subject}\n"
        sender = f"From: {msg.sender}\n"
        to = f"To: {msg.to}\n"
        cc = f"Cc: {msg.cc}\n" if msg.cc else ""
        bcc = f"Bcc: {msg.bcc}\n" if msg.bcc else ""
        body = f"\n{msg.body}"

        # Combine them into a single text string
        msg_text = subject + sender + to + cc + bcc + body
        return msg_text
