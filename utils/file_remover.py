# 0) Colab 인증
from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

service = build('drive', 'v3')

def _escape_for_q(s: str) -> str:
    # Drive 'q'에서 문자열 리터럴은 작은따옴표로 감싸며,
    # 내부의 작은따옴표는 \'
    return s.replace("'", "\\'")

def _get_id_by_path(path: str) -> str:
    """
    /content/drive/MyDrive/... 경로를 따라 내려가며 대상의 fileId를 찾는다.
    동일 이름이 여러 개면 첫 번째 것을 사용.
    (중간 경로는 폴더를 우선 선택)
    """
    if not path.startswith("/content/drive/MyDrive/"):
        raise ValueError("MyDrive 경로만 지원합니다: /content/drive/MyDrive/...")

    parts = [p for p in path.split("/") if p][3:]  # ['폴더', '파일'] 식으로
    if not parts:
        raise ValueError("삭제 대상 파일/폴더 이름이 비었습니다.")

    parent_id = "root"
    for i, name in enumerate(parts):
        name_q = _escape_for_q(name)
        q = f"name = '{name_q}' and '{parent_id}' in parents and trashed = false"
        resp = service.files().list(q=q, fields="files(id,name,mimeType)").execute()
        hits = resp.get("files", [])
        if not hits:
            raise FileNotFoundError(f"경로를 찾을 수 없음: {'/'.join(parts[:i+1])}")

        # 중간 단계는 폴더 우선, 마지막은 첫 항목 사용
        if i < len(parts) - 1:
            folder = next((f for f in hits if f.get("mimeType") == "application/vnd.google-apps.folder"), None)
            target = folder or hits[0]
        else:
            target = hits[0]

        parent_id = target["id"]

    return parent_id

def delete_permanently_by_path(path: str):
    """경로로 대상 찾아 Drive API delete → 영구 삭제"""
    if not path.startswith("/content/drive/MyDrive/") :
        os.remove(path)
        return
    else :
        file_id = _get_id_by_path(path)
        service.files().delete(fileId=file_id).execute()
        print("Permanently deleted:", path)

def delete_permanently_by_id(file_id: str):
    """fileId를 이미 아는 경우 바로 영구 삭제"""
    service.files().delete(fileId=file_id).execute()
    print("Permanently deleted (by id):", file_id)
