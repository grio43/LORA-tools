import pyarrow.parquet as pq

# Make sure to use the correct path to your metadata file
schema = pq.read_schema(r"J:\New file\Danbooru2004\metadata.parquet")
print(schema.names)