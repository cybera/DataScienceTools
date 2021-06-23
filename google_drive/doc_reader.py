import os.path

from apiclient import errors
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from oauth2client import client, file


SCOPES = ['https://www.googleapis.com/auth/documents.readonly',
          'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/drive.metadata.readonly'] #


def get_credentials():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def traverse_folder(service, folder_id, docs_service, path):
    folder = service.files().get(fileId = folder_id).execute()
    results = service.files().list(q=f"'{folder_id}' in parents", pageSize=100, fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])
    path = path+folder['name']+"/"
    # print(f"{path}")
    if not items:
        print('No files found.')
    else:
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.document':
                # print(f"found document {item['id']}")
                read_file(docs_service, item['id'], path)
            elif item['mimeType'] == 'application/vnd.google-apps.folder':
                traverse_folder(service, item['id'], docs_service, path)


def read_file(docs_service, document_id, path):
    doc = docs_service.documents().get(documentId=document_id).execute()
    print(f"\n[{path}{doc['title']}]:")
    # print(f"{doc}")
    doc_content = doc.get('body').get('content')
    content = read_strucutural_elements(doc_content)
    # print(content)
    for string in content.splitlines():
        if not string.strip():
            continue
        print(string)


def read_strucutural_elements(elements):
    """Recurses through a list of Structural Elements to read a document's text where text may be
        in nested elements.

        Args:
            elements: a list of Structural Elements.
    """
    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                text += read_paragraph_element(elem)
        elif 'table' in value:
            # The text in table cells are in nested Structural Elements and tables may be
            # nested.
            table = value.get('table')
            for row in table.get('tableRows'):
                cells = row.get('tableCells')
                for cell in cells:
                    text += read_strucutural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text in the TOC is also in a Structural Element.
            toc = value.get('tableOfContents')
            text += read_strucutural_elements(toc.get('content'))
    return text


def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get('textRun')
    if not text_run:
        return ''
    content = text_run.get('content')

    return content


def main(root_folder_id):
    creds = get_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)
    traverse_folder(drive_service, root_folder_id, docs_service,"...")




if __name__ == '__main__':
    root_folder_id = "1zrufPTC7NidVTGV0ZZP8w0P_-yI6ce9b"
    root_folder_id = "0BxvQDiB_Bsb2R091UGtvLTNvbXM"
    root_folder_id = "0B57YUfZ-xUQbWG92NHhKZUhHaFE"

    main(root_folder_id)
