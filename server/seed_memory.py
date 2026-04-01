import sys
import os

from pipeline import DebugPrioritizer, add_to_memory

def seed():
    print("Running pipeline to generate a sample cluster vector...")
    dp = DebugPrioritizer()
    dp.ingest()
    dp.embed_and_cluster()

    if not dp.cluster_centroids:
        print("No clusters formed. Exiting.")
        sys.exit(1)

    # Let's forcefully pick one of the prominent clusters to seed
    cid = max(dp.cluster_centroids.keys())
    if cid == -1:
        cid = max([k for k in dp.cluster_centroids.keys() if k != -1])
        
    vec = dp.cluster_centroids[cid]
    sig = dp.signatures.get(cid, "UVM_ERROR: some sample signature")
    
    # Store it with the requested demo text
    add_to_memory(
        centroid=vec,
        signature=sig,
        project_name="Apollo_V2",
        previous_fix_note="changing axi_master.sv line 88"
    )
    print("Project memory successfully seeded with vector shape:", vec.shape)

if __name__ == "__main__":
    seed()
