import sys
from pymilvus import MilvusClient, MilvusException

def clear_milvus(uri="http://localhost:19530"):
    print(f"[*] Dang ket noi toi Milvus tai {uri} de xoa du lieu...")
    try:
        client = MilvusClient(uri=uri)
        collections = client.list_collections()
        
        if not collections:
            print("[+] Database hien khong co collection nao de xoa.")
            return

        for coll_name in collections:
            print(f"[*] Dang xoa collection: '{coll_name}'...")
            client.drop_collection(collection_name=coll_name)
            print(f"[+] Da xoa thanh cong collection '{coll_name}'.")
            
        print("[+] Xoa toan bo du lieu trong Milvus thanh cong!")
        
    except MilvusException as e:
        print("[-] Khong the ket noi toi Milvus.")
        print(f"Chi tiet loi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Da xay ra loi khong xac dinh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    confirm = input("Ban co chac chan muon xoa TOAN BO du lieu trong Vector Database khong? (y/n): ")
    if confirm.lower() == 'y':
        clear_milvus()
    else:
        print("Da huy thao tac xoa.")

# Xóa toàn bộ dữ liệu trong Milvus
# .\.venv\Scripts\python milvus\clear_milvus.py