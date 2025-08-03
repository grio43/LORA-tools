from pyarrow import dataset as ds, parquet as pq

SRC = "tables"                   # folder with the shards
DST = "rule34_full.parquet"      # output file

# Arrow’s dataset API streams in record batches (no OOM)
source = ds.dataset(SRC, format="parquet")
writer = None

for batch in source.to_batches():
    if writer is None:
        # Create the writer on first batch so we inherit the schema
        writer = pq.ParquetWriter(DST, batch.schema, compression="zstd")
    writer.write_table(batch)

writer.close()
print("✅ Wrote", DST)
