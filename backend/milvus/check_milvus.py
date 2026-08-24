import sys
from pymilvus import MilvusClient, MilvusException

def check_milvus(uri="http://localhost:19530"):
    print(f"[*] Dang kiem tra ket noi toi Milvus tai {uri}...")
    try:
        client = MilvusClient(uri=uri)
        collections = client.list_collections()
        
        print("[+] Ket noi Milvus thanh cong va dang hoat dong binh thuong!")
        print(f"[*] Cac collection hien co: {collections}\n")
        
        if not collections:
            print("[-] Database hien chua co collection nao.")
        else:
            for coll_name in collections:
                print(f"--- Thong tin Collection: '{coll_name}' ---")
                
                # Cần load collection trước khi query hoặc lấy thông tin chi tiết (tùy version, nhưng làm cho chắc)
                try:
                    client.load_collection(collection_name=coll_name)
                    
                    # Lấy số lượng entity
                    stats = client.get_collection_stats(collection_name=coll_name)
                    row_count = stats.get("row_count", 0)
                    print(f"[*] So luong chunk (row_count): {row_count}")
                    
                    # Truy vấn thử để lấy danh sách tên tài liệu (nếu có trường document_name)
                    if int(row_count) > 0:
                        try:
                            # Lấy mẫu 1000 chunk để lọc ra các tên tài liệu độc nhất
                            # expr="" means all in some versions, but usually we need an expr. 
                            # id >= "" is a dummy expression that usually works if id is varchar
                            res = client.query(
                                collection_name=coll_name,
                                filter="id != ''",
                                output_fields=["document_name"],
                                limit=1000
                            )
                            doc_names = set([r.get("document_name") for r in res if r.get("document_name")])
                            if doc_names:
                                print(f"[*] Cac tai lieu (document) da duoc luu: {list(doc_names)}")
                            else:
                                print("[*] Khong the trich xuat ten tai lieu (co the thieu truong document_name).")
                        except Exception as e:
                            print(f"[-] Loi khi truy van danh sach tai lieu: {e}")
                except Exception as e:
                    print(f"[-] Loi khi thong ke collection '{coll_name}': {e}")
                print("-" * 40)
                
    except MilvusException as e:
        print("[-] Khong the ket noi toi Milvus.")
        print("Vui long kiem tra xem container cua Milvus da duoc khoi chay chua (vi du: dung lenh 'docker ps' hoac 'docker-compose up -d').")
        print(f"Chi tiet loi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Da xay ra loi khong xac dinh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_milvus()

# Kiểm tra số lượng chunk và danh sách tên file trong DB
#.\.venv\Scripts\python milvus\check_milvus.py