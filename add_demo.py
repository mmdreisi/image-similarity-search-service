from db.read_sql import read_templates_from_sqlite
from features.extractor_dino_v2 import FeatureExtractor
from db.vector_store import VectorStore

db_path = r"E:\work\check_them\data\temp_data\templates.db"
table_name  = 'templates'

data = read_templates_from_sqlite(db_path=db_path , table_name=table_name)
extractor = FeatureExtractor()
vector_s = VectorStore()
#print(data)
for template in data:
    
    vector = extractor.extract_from_path(template['image_path'])
    vector_s.add_item(vector , template)
    
vector_s.save()

print('every thing is good!!!!!')
    