#!/usr/bin/env python3
"""
Google Docs PDF Generator - Batch Version for Storage Quota Management

This script processes files in small batches to avoid storage quota issues:
1. Processes only a few files at a time
2. Immediately cleans up after each conversion
3. Includes storage quota monitoring
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


class GoogleDocsConverterBatch:
    def __init__(self, credentials_path: str):
        """Initialize the Google Drive API client."""
        self.credentials_path = credentials_path
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Google Drive API using service account."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        return service
    
    def get_storage_info(self):
        """Get current storage usage information."""
        try:
            about = self.service.about().get(fields='storageQuota').execute()
            quota = about.get('storageQuota', {})
            
            limit = int(quota.get('limit', 0))
            usage = int(quota.get('usage', 0))
            
            return {
                'limit_gb': limit / (1024**3),
                'usage_gb': usage / (1024**3),
                'available_gb': (limit - usage) / (1024**3),
                'usage_percent': (usage / limit * 100) if limit > 0 else 0
            }
        except Exception as e:
            print(f"Warning: Could not get storage info: {e}")
            return None
    
    def convert_single_file(self, docx_path: str, pdf_path: str) -> bool:
        """
        Convert a single file with immediate cleanup to minimize storage usage.
        """
        file_name = Path(docx_path).stem
        
        try:
            print(f"  → Uploading {file_name}...")
            
            # Upload and convert to Google Docs
            file_metadata = {
                'name': f"temp_{file_name}",
                'mimeType': 'application/vnd.google-apps.document'
            }
            
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
            
            file_id = file.get('id')
            print(f"  → Converting to PDF...")
            
            # Small delay to ensure conversion is complete
            time.sleep(1)
            
            # Export as PDF
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType='application/pdf'
            )
            
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
            
            # Write to local file
            with open(pdf_path, 'wb') as f:
                f.write(file_buffer.getvalue())
            
            print(f"  → Cleaning up temporary file...")
            
            # Immediately delete the temporary Google Doc
            self.service.files().delete(fileId=file_id).execute()
            
            print(f"  ✓ Successfully converted {file_name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error converting {file_name}: {str(e)}")
            # Try to clean up if file was created
            try:
                if 'file_id' in locals():
                    self.service.files().delete(fileId=file_id).execute()
            except:
                pass
            return False
    
    def empty_trash(self):
        """Empty the Google Drive trash to free up storage."""
        try:
            print("Emptying Google Drive trash...")
            self.service.files().emptyTrash().execute()
            print("✓ Trash emptied")
        except Exception as e:
            print(f"Warning: Could not empty trash: {e}")


def batch_convert_with_storage_management(credentials_path: str, 
                                        docx_folder: str, 
                                        pdf_folder: str,
                                        batch_size: int = 5):
    """
    Convert files in small batches to manage storage quota.
    """
    converter = GoogleDocsConverterBatch(credentials_path)
    
    # Create output folder
    Path(pdf_folder).mkdir(exist_ok=True)
    
    # Check initial storage
    storage_info = converter.get_storage_info()
    if storage_info:
        print(f"Storage: {storage_info['usage_gb']:.1f}GB / {storage_info['limit_gb']:.1f}GB "
              f"({storage_info['usage_percent']:.1f}% used)")
        
        if storage_info['usage_percent'] > 90:
            print("⚠ Warning: Storage is over 90% full!")
            converter.empty_trash()
    
    # Get all .docx files that need conversion
    docx_files = []
    for docx_file in Path(docx_folder).glob("*.docx"):
        pdf_file = Path(pdf_folder) / f"{docx_file.stem}.pdf"
        if not pdf_file.exists():
            docx_files.append(docx_file)
    
    total_files = len(docx_files)
    print(f"\nFound {total_files} files to convert")
    
    if total_files == 0:
        print("All files already converted!")
        return
    
    successful = 0
    failed = 0
    
    # Process in batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = docx_files[batch_start:batch_end]
        
        print(f"\n=== Batch {batch_start//batch_size + 1}: "
              f"Files {batch_start + 1}-{batch_end} of {total_files} ===")
        
        for i, docx_file in enumerate(batch_files):
            file_num = batch_start + i + 1
            pdf_file = Path(pdf_folder) / f"{docx_file.stem}.pdf"
            
            print(f"[{file_num}/{total_files}] Processing {docx_file.name}")
            
            success = converter.convert_single_file(str(docx_file), str(pdf_file))
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Delay between files
            time.sleep(0.5)
        
        # Empty trash between batches
        if batch_end < total_files:
            print(f"\nCompleted batch {batch_start//batch_size + 1}")
            converter.empty_trash()
            
            # Check storage after batch
            storage_info = converter.get_storage_info()
            if storage_info:
                print(f"Storage after batch: {storage_info['usage_gb']:.1f}GB "
                      f"({storage_info['usage_percent']:.1f}% used)")
            
            print("Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    # Final cleanup
    converter.empty_trash()
    
    print(f"\n=== Conversion Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {total_files}")


def main():
    """Main execution function."""
    CREDENTIALS_PATH = "google_credentials.json"
    DOCX_FOLDER = "wikipedia_docs"
    PDF_FOLDER = "google_docs_pdfs"
    BATCH_SIZE = 3  # Process only 3 files at a time
    
    if not Path(CREDENTIALS_PATH).exists():
        print(f"Error: Credentials file not found: {CREDENTIALS_PATH}")
        return
    
    if not Path(DOCX_FOLDER).exists():
        print(f"Error: DOCX folder not found: {DOCX_FOLDER}")
        return
    
    print("Starting batch conversion with storage management...")
    batch_convert_with_storage_management(
        CREDENTIALS_PATH, DOCX_FOLDER, PDF_FOLDER, BATCH_SIZE
    )


if __name__ == "__main__":
    main()