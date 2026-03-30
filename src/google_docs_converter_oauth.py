#!/usr/bin/env python3
"""
Google Docs PDF Generator - OAuth Version (Uses Your Personal Google Drive)

This script uses OAuth to authenticate with YOUR personal Google account:
1. One-time browser login to authorize the app
2. Uses your personal 15GB Google Drive storage
3. Processes files in batches to manage storage
4. More reliable than service account approach
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import List, Tuple

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io


class GoogleDocsConverterOAuth:
    def __init__(self, credentials_path: str):
        """Initialize the Google Drive API client with OAuth."""
        self.credentials_path = credentials_path
        self.token_path = "token.pickle"
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Google Drive API using OAuth."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        
        creds = None
        
        # Check for existing token
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                print("Setting up OAuth authentication...")
                print("A browser window will open for you to authorize the app.")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
                print("✓ Credentials saved for future use")
        
        service = build('drive', 'v3', credentials=creds)
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
    
    def create_temp_folder(self) -> str:
        """Create a temporary folder in Google Drive."""
        file_metadata = {
            'name': 'ForensicsDetective_PDF_Conversion_Temp',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        folder_id = file.get('id')
        print(f"✓ Created temporary folder in your Google Drive")
        return folder_id
    
    def convert_single_file(self, docx_path: str, pdf_path: str, folder_id: str = None) -> bool:
        """Convert a single file with immediate cleanup."""
        file_name = Path(docx_path).stem
        
        try:
            print(f"  → Uploading {file_name}...")
            
            # Upload and convert to Google Docs
            file_metadata = {
                'name': f"temp_{file_name}",
                'mimeType': 'application/vnd.google-apps.document'
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
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
            
            print(f"  → Cleaning up...")
            
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
    
    def delete_folder(self, folder_id: str):
        """Delete a folder from Google Drive."""
        try:
            self.service.files().delete(fileId=folder_id).execute()
            print("✓ Cleaned up temporary folder")
        except Exception as e:
            print(f"Warning: Could not delete temporary folder: {e}")
    
    def empty_trash(self):
        """Empty the Google Drive trash."""
        try:
            print("Emptying Google Drive trash...")
            self.service.files().emptyTrash().execute()
            print("✓ Trash emptied")
        except Exception as e:
            print(f"Warning: Could not empty trash: {e}")


def oauth_batch_convert(credentials_path: str, 
                       docx_folder: str, 
                       pdf_folder: str,
                       batch_size: int = 5):
    """Convert files using OAuth authentication."""
    
    print("🚀 Starting OAuth-based PDF conversion")
    print("This will use your personal Google Drive storage")
    
    converter = GoogleDocsConverterOAuth(credentials_path)
    
    # Create output folder
    Path(pdf_folder).mkdir(exist_ok=True)
    
    # Check storage
    storage_info = converter.get_storage_info()
    if storage_info:
        print(f"\n📊 Your Google Drive Storage:")
        print(f"Used: {storage_info['usage_gb']:.1f}GB / {storage_info['limit_gb']:.1f}GB "
              f"({storage_info['usage_percent']:.1f}%)")
        print(f"Available: {storage_info['available_gb']:.1f}GB")
        
        if storage_info['available_gb'] < 1.0:
            print("⚠️  Warning: Less than 1GB available - consider freeing up space first!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Create temporary folder
    temp_folder_id = converter.create_temp_folder()
    
    # Get files to convert
    docx_files = []
    for docx_file in Path(docx_folder).glob("*.docx"):
        pdf_file = Path(pdf_folder) / f"{docx_file.stem}.pdf"
        if not pdf_file.exists():
            docx_files.append(docx_file)
    
    total_files = len(docx_files)
    print(f"\n📁 Found {total_files} files to convert")
    
    if total_files == 0:
        print("🎉 All files already converted!")
        converter.delete_folder(temp_folder_id)
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
            
            success = converter.convert_single_file(
                str(docx_file), str(pdf_file), temp_folder_id)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Small delay between files
            time.sleep(0.3)
        
        # Status update between batches
        if batch_end < total_files:
            print(f"\n✅ Completed batch {batch_start//batch_size + 1}")
            converter.empty_trash()
            
            # Check storage after batch
            storage_info = converter.get_storage_info()
            if storage_info:
                print(f"📊 Storage: {storage_info['usage_gb']:.1f}GB "
                      f"({storage_info['usage_percent']:.1f}% used)")
            
            print("⏱️  Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    # Final cleanup
    print(f"\n🧹 Final cleanup...")
    converter.delete_folder(temp_folder_id)
    converter.empty_trash()
    
    print(f"\n🎉 Conversion Complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {total_files}")
    
    if failed > 0:
        print(f"\n💡 You can re-run the script to retry failed conversions")


def main():
    """Main execution function."""
    CREDENTIALS_PATH = "oauth_credentials.json"  # Different name for OAuth
    DOCX_FOLDER = "wikipedia_docs"
    PDF_FOLDER = "google_docs_pdfs"
    BATCH_SIZE = 5  # Process 5 files at a time
    
    if not Path(CREDENTIALS_PATH).exists():
        print(f"❌ Error: OAuth credentials file not found: {CREDENTIALS_PATH}")
        print("\n📋 Please follow the OAuth setup guide:")
        print("1. Download OAuth credentials from Google Cloud Console")
        print("2. Rename to 'oauth_credentials.json'")
        print("3. Place in this directory")
        return
    
    if not Path(DOCX_FOLDER).exists():
        print(f"❌ Error: DOCX folder not found: {DOCX_FOLDER}")
        return
    
    oauth_batch_convert(CREDENTIALS_PATH, DOCX_FOLDER, PDF_FOLDER, BATCH_SIZE)


if __name__ == "__main__":
    main()