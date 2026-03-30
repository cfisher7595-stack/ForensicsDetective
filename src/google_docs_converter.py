#!/usr/bin/env python3
"""
Google Docs PDF Generator for Forensics Detective Project

This script automates the conversion of Word documents to PDFs via Google Docs:
1. Uploads .docx files to Google Drive
2. Converts them to Google Docs format
3. Exports as PDFs and downloads them locally

Requirements:
- Google Cloud Project with Drive API enabled
- Service account credentials JSON file
- google-api-python-client and google-auth packages
"""

import os
import time
import json
from pathlib import Path
from typing import List, Tuple

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io


class GoogleDocsConverter:
    def __init__(self, credentials_path: str):
        """Initialize the Google Drive API client."""
        self.credentials_path = credentials_path
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Google Drive API using service account."""
        # Define the scopes
        SCOPES = ['https://www.googleapis.com/auth/drive']
        
        # Load credentials
        credentials = Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES)
        
        # Build the service
        service = build('drive', 'v3', credentials=credentials)
        return service
    
    def upload_and_convert_docx(self, docx_path: str, folder_id: str = None) -> str:
        """
        Upload a .docx file and convert it to Google Docs format.
        
        Args:
            docx_path: Path to the .docx file
            folder_id: Optional Google Drive folder ID to upload to
            
        Returns:
            File ID of the uploaded Google Doc
        """
        file_name = Path(docx_path).stem
        
        # File metadata
        file_metadata = {
            'name': file_name,
            'mimeType': 'application/vnd.google-apps.document'  # Convert to Google Docs
        }
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Upload and convert
        media = MediaFileUpload(
            docx_path,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            resumable=True
        )
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    
    def export_as_pdf(self, file_id: str, output_path: str):
        """
        Export a Google Doc as PDF and save locally.
        
        Args:
            file_id: Google Drive file ID of the document
            output_path: Local path to save the PDF
        """
        request = self.service.files().export_media(
            fileId=file_id,
            mimeType='application/pdf'
        )
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        
        while done is False:
            status, done = downloader.next_chunk()
        
        # Write to local file
        with open(output_path, 'wb') as f:
            f.write(file.getvalue())
    
    def delete_file(self, file_id: str):
        """Delete a file from Google Drive."""
        self.service.files().delete(fileId=file_id).execute()
    
    def create_folder(self, folder_name: str) -> str:
        """Create a folder in Google Drive and return its ID."""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')
    
    def convert_docx_to_pdf(self, docx_path: str, pdf_path: str, 
                           cleanup: bool = True, folder_id: str = None):
        """
        Complete workflow: upload docx, convert to Google Docs, export as PDF.
        
        Args:
            docx_path: Path to input .docx file
            pdf_path: Path for output .pdf file
            cleanup: Whether to delete the Google Doc after conversion
            folder_id: Optional Google Drive folder to use for temporary storage
        """
        print(f"Converting {Path(docx_path).name}...")
        
        try:
            # Upload and convert to Google Docs
            file_id = self.upload_and_convert_docx(docx_path, folder_id)
            
            # Small delay to ensure conversion is complete
            time.sleep(1)
            
            # Export as PDF
            self.export_as_pdf(file_id, pdf_path)
            
            # Clean up the Google Doc if requested
            if cleanup:
                self.delete_file(file_id)
            
            print(f"✓ Successfully converted {Path(docx_path).name}")
            return True
            
        except Exception as e:
            print(f"✗ Error converting {Path(docx_path).name}: {str(e)}")
            return False


def batch_convert_documents(credentials_path: str, 
                          docx_folder: str, 
                          pdf_folder: str,
                          create_temp_folder: bool = True):
    """
    Convert all .docx files in a folder to PDFs via Google Docs.
    
    Args:
        credentials_path: Path to Google service account credentials JSON
        docx_folder: Folder containing .docx files
        pdf_folder: Folder to save generated PDFs
        create_temp_folder: Whether to create a temporary folder in Google Drive
    """
    # Initialize converter
    converter = GoogleDocsConverter(credentials_path)
    
    # Create output folder if it doesn't exist
    Path(pdf_folder).mkdir(exist_ok=True)
    
    # Create temporary folder in Google Drive if requested
    temp_folder_id = None
    if create_temp_folder:
        temp_folder_id = converter.create_folder("ForensicsDetective_Temp")
        print(f"Created temporary Google Drive folder: {temp_folder_id}")
    
    # Get all .docx files
    docx_files = list(Path(docx_folder).glob("*.docx"))
    total_files = len(docx_files)
    
    print(f"Found {total_files} .docx files to convert")
    
    successful = 0
    failed = 0
    
    for i, docx_file in enumerate(docx_files, 1):
        pdf_file = Path(pdf_folder) / f"{docx_file.stem}.pdf"
        
        print(f"\n[{i}/{total_files}] Processing {docx_file.name}")
        
        if pdf_file.exists():
            print(f"⚠ PDF already exists, skipping: {pdf_file.name}")
            continue
        
        success = converter.convert_docx_to_pdf(
            str(docx_file), 
            str(pdf_file),
            cleanup=True,
            folder_id=temp_folder_id
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Small delay between files to be respectful to the API
        if i < total_files:
            time.sleep(0.5)
    
    # Clean up temporary folder
    if temp_folder_id:
        try:
            converter.delete_file(temp_folder_id)
            print(f"\nCleaned up temporary Google Drive folder")
        except Exception as e:
            print(f"Warning: Could not delete temporary folder: {e}")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {total_files}")


def main():
    """Main execution function."""
    # Configuration
    CREDENTIALS_PATH = "google_credentials.json"  # You'll need to create this
    DOCX_FOLDER = "wikipedia_docs"
    PDF_FOLDER = "google_docs_pdfs"
    
    # Check if credentials file exists
    if not Path(CREDENTIALS_PATH).exists():
        print(f"Error: Credentials file not found: {CREDENTIALS_PATH}")
        print("\nTo set up Google Drive API credentials:")
        print("1. Go to Google Cloud Console (console.cloud.google.com)")
        print("2. Create a new project or select existing one")
        print("3. Enable the Google Drive API")
        print("4. Create a Service Account")
        print("5. Download the JSON credentials file")
        print("6. Rename it to 'google_credentials.json' in this directory")
        return
    
    if not Path(DOCX_FOLDER).exists():
        print(f"Error: DOCX folder not found: {DOCX_FOLDER}")
        return
    
    # Run the batch conversion
    batch_convert_documents(CREDENTIALS_PATH, DOCX_FOLDER, PDF_FOLDER)


if __name__ == "__main__":
    main()