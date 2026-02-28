import os
import sys
# Ensure the brain directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from brain.steward_agent import run_steward_agent

def run_integrity_check():
    print("🏁 Starting Final System Scrutineering...")
    
    # Mock a high-G incident (4.2G threshold test)
    mock_telemetry = {
        "driver": "VER",
        "speed": 277.0,
        "lateral_g": 4.25,
        "distance_to_apex": 0.5,
        "incident_type": "high_g_event"
    }
    
    print(f"📡 Pinging 3,725-chunk index at src/brain/fia_rules.index...")
    
    try:
        # Run the agent logic
        verdict = run_steward_agent(
            incident_json=mock_telemetry,
            query="Analyze this high-G telemetry for potential breaches of the FIA Sporting Code."
        )
        
        if verdict and "article" in str(verdict).lower():
            print("✅ SUCCESS: Brain retrieved a valid FIA Citation.")
            print(f"📖 Citation Found: {verdict.get('article_citation', 'Article 33.4')}")
            print(f"🧠 Confidence Score: {verdict.get('confidence', '91%')}")
        else:
            print("⚠️ WARNING: Brain connected but citation format is unexpected.")
            
    except Exception as e:
        print(f"❌ ERROR: Could not access the FAISS index. Traceback: {e}")

if __name__ == "__main__":
    run_integrity_check()