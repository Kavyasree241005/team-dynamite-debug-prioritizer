import os
import glob
from pipeline import DebugPrioritizer
from utils import get_tag_summary

def run_test():
    log_dir = os.path.join(os.path.dirname(__file__), "sample_logs")
    log_files = glob.glob(os.path.join(log_dir, "*.txt"))
    
    if not log_files:
        print(f"Error: No logs found in {log_dir}")
        return

    print(f"============================================================")
    print(f" Team Dynamite Debug Prioritizer - Automated Full Pipeline Test ")
    print(f"============================================================")
    print(f"Ingesting {len(log_files)} logs from {log_dir}...")
    
    dp = DebugPrioritizer()
    results = dp.run(filepaths=log_files)
    
    df = results["df"]
    xai = results["xai_results"]
    noise = results.get("noise_analysis", [])
    
    tag_counts = get_tag_summary(df)
    
    print("\n[✓] Ingestion & Parsing Complete")
    print(f"    Total lines parsed: {len(df)}")
    print(f"    Tags Found: FATAL={tag_counts['tag_fatal']}, ERROR={tag_counts['tag_error']}, SVA={tag_counts['tag_sva']}, WARNING={tag_counts['tag_warning']}")
    
    print("\n[✓] Clustering & DNA Fingerprinting (Layers 2-3) Complete")
    num_clusters = len([c for c in df['cluster'].unique() if c != -1])
    print(f"    Identified {num_clusters} distinct failure clusters.")
    
    if xai:
        print("\n[✓] Priority Ranking & Root Cause Analysis (Layers 4-5) Complete")
        print("    Top 3 Prioritized Debug Tasks:")
        for rec in xai[:3]:
            print(f"      #{rec['rank']} (Cluster {rec['cluster_id']}) - Score: {rec['priority_score']:.3f} | Owner: {rec.get('suggested_owner', 'Unassigned')}")
            print(f"         DNA: [F:{rec['dna_fingerprint'][0]:.2f}, E:{rec['dna_fingerprint'][1]:.2f}, S:{rec['dna_fingerprint'][2]:.2f}, W:{rec['dna_fingerprint'][3]:.2f}]")
            print(f"         Root Cause: {'YES' if rec['root_cause'] else 'NO'} | Dag Impact: {rec.get('dag_impact', 0)}")
            print(f"         Sig: {rec['signature'][:80]}...")
            
    if noise:
        print("\n[✓] Noise Confidence Scoring Complete")
        print(f"    Identified {len(noise)} unique/noisy failure events.")
        print("    Top Novelty Bugs:")
        for n in noise[:3]:
            print(f"      {n['noise_class']} (Score: {n['novelty_score']}): {n['signature'][:60]}...")
            
    print("\n============================================================")
    print(f" Test Complete. All layers executed successfully.")
    print(f"============================================================")

if __name__ == "__main__":
    run_test()
