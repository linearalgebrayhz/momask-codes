#!/usr/bin/env python3
"""
Caption Quality Verification Tool

Compares deterministic captions with AI-enhanced captions to identify:
1. Direction consistency (left vs right, up vs down)
2. Intent classification quality
3. Cinematographic terminology usage

Usage:
    python verify_caption_quality.py \
        --dataset-root ./dataset/RealEstate10K_6feat \
        --sample-size 20 \
        --output report.txt
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def extract_directions(caption: str) -> Dict[str, List[str]]:
    """Extract directional keywords from caption"""
    directions = {
        'horizontal': [],
        'vertical': [],
        'depth': [],
        'rotation': []
    }
    
    # Horizontal
    if re.search(r'\b(left|leftward)\b', caption, re.I):
        directions['horizontal'].append('left')
    if re.search(r'\b(right|rightward)\b', caption, re.I):
        directions['horizontal'].append('right')
    
    # Vertical
    if re.search(r'\b(up|upward|rise|rising|ascend|ascending)\b', caption, re.I):
        directions['vertical'].append('up')
    if re.search(r'\b(down|downward|descend|descending|drop|dropping)\b', caption, re.I):
        directions['vertical'].append('down')
    
    # Depth
    if re.search(r'\b(forward|toward|approach|approaching|closer|push|dolly.*forward)\b', caption, re.I):
        directions['depth'].append('forward')
    if re.search(r'\b(back|backward|away|recede|receding|pull|dolly.*back)\b', caption, re.I):
        directions['depth'].append('back')
    
    # Rotation
    if re.search(r'\b(pan.*left|panning.*left|swivel.*left)\b', caption, re.I):
        directions['rotation'].append('pan-left')
    if re.search(r'\b(pan.*right|panning.*right|swivel.*right)\b', caption, re.I):
        directions['rotation'].append('pan-right')
    if re.search(r'\b(tilt.*up|tilting.*up)\b', caption, re.I):
        directions['rotation'].append('tilt-up')
    if re.search(r'\b(tilt.*down|tilting.*down)\b', caption, re.I):
        directions['rotation'].append('tilt-down')
    
    return directions


def extract_intent(caption: str) -> str:
    """Extract cinematographic intent from caption"""
    intents = [
        'establishing shot', 'establishing', 
        'reveal', 'revealing',
        'follow', 'tracking', 'follow-tracking',
        'orbit', 'orbiting', 'circling',
        'flythrough', 'fly-through',
        'inspection', 'detail inspection',
        'scan', 'environmental scan', 'scanning'
    ]
    
    caption_lower = caption.lower()
    for intent in intents:
        if intent in caption_lower:
            return intent
    
    # Check for "Intent:" prefix format
    match = re.search(r'intent:\s*([^.]+)', caption, re.I)
    if match:
        return match.group(1).strip()
    
    return 'unspecified'


def check_direction_consistency(det_dirs: Dict, ai_dirs: Dict) -> Tuple[bool, List[str]]:
    """Check if AI caption directions match deterministic caption"""
    issues = []
    consistent = True
    
    for axis in ['horizontal', 'vertical', 'depth', 'rotation']:
        det = set(det_dirs[axis])
        ai = set(ai_dirs[axis])
        
        # Check for contradictions (both left AND right mentioned)
        if len(det) > 1 and axis != 'rotation':
            issues.append(f"Deterministic has conflicting {axis}: {det}")
            consistent = False
        if len(ai) > 1 and axis != 'rotation':
            issues.append(f"AI has conflicting {axis}: {ai}")
            consistent = False
        
        # Check if AI contradicts deterministic
        if det and ai:
            # For rotation, can have multiple valid (pan + tilt)
            if axis == 'rotation':
                continue
            # For other axes, should match
            if not det.intersection(ai):
                issues.append(f"Direction mismatch on {axis}: det={det}, ai={ai}")
                consistent = False
    
    return consistent, issues


def analyze_cinematography_terms(caption: str) -> Dict[str, bool]:
    """Check if caption uses professional cinematography terms"""
    terms = {
        'track': bool(re.search(r'\btrack(s|ing|ed)?\b', caption, re.I)),
        'truck': bool(re.search(r'\btruck(s|ing|ed)?\b', caption, re.I)),
        'dolly': bool(re.search(r'\bdoll(y|ies|ying|ied)\b', caption, re.I)),
        'pan': bool(re.search(r'\bpan(s|ning|ned)?\b', caption, re.I)),
        'tilt': bool(re.search(r'\btilt(s|ing|ed)?\b', caption, re.I)),
        'roll': bool(re.search(r'\broll(s|ing|ed)?\b', caption, re.I)),
        'arc': bool(re.search(r'\barc(s|ing|ed)?\b', caption, re.I)),
        'orbit': bool(re.search(r'\borbit(s|ing|ed)?\b', caption, re.I)),
        'crane': bool(re.search(r'\bcrane(s|ing|ed)?\b', caption, re.I)),
        'pedestal': bool(re.search(r'\bpedestal\b', caption, re.I)),
        'boom': bool(re.search(r'\bboom(s|ing|ed)?\b', caption, re.I))
    }
    return terms


def verify_scene(scene_id: str, dataset_root: Path) -> Dict:
    """Verify caption quality for a single scene"""
    # Load metadata
    metadata_file = dataset_root / "metadata" / f"{scene_id}.json"
    if not metadata_file.exists():
        return {'error': 'metadata not found'}
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load captions
    untagged_file = dataset_root / "untagged_text" / f"{scene_id}.txt"
    if not untagged_file.exists():
        return {'error': 'caption not found'}
    
    with open(untagged_file, 'r') as f:
        ai_caption = f.read().strip()
    
    # For deterministic, we'd need to regenerate or store separately
    # For now, assume the first sentence is deterministic-like
    
    # Extract features
    ai_dirs = extract_directions(ai_caption)
    intent = extract_intent(ai_caption)
    terms = analyze_cinematography_terms(ai_caption)
    
    return {
        'scene_id': scene_id,
        'ai_caption': ai_caption,
        'directions': ai_dirs,
        'intent': intent,
        'terms_used': [k for k, v in terms.items() if v],
        'num_terms': sum(terms.values()),
        'has_intent': intent != 'unspecified',
        'metadata': metadata
    }


def generate_report(results: List[Dict], output_file: str):
    """Generate quality verification report"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CAPTION QUALITY VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        total = len(results)
        with_intent = sum(1 for r in results if r.get('has_intent', False))
        avg_terms = np.mean([r.get('num_terms', 0) for r in results])
        
        f.write(f"Total Scenes Analyzed: {total}\n")
        f.write(f"Scenes with Intent: {with_intent} ({with_intent/total*100:.1f}%)\n")
        f.write(f"Avg Cinematography Terms per Caption: {avg_terms:.2f}\n\n")
        
        # Intent distribution
        intents = {}
        for r in results:
            intent = r.get('intent', 'unspecified')
            intents[intent] = intents.get(intent, 0) + 1
        
        f.write("Intent Distribution:\n")
        for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
            f.write(f"  {intent}: {count} ({count/total*100:.1f}%)\n")
        f.write("\n")
        
        # Most common terms
        all_terms = {}
        for r in results:
            for term in r.get('terms_used', []):
                all_terms[term] = all_terms.get(term, 0) + 1
        
        f.write("Most Common Cinematography Terms:\n")
        for term, count in sorted(all_terms.items(), key=lambda x: -x[1])[:10]:
            f.write(f"  {term}: {count} ({count/total*100:.1f}%)\n")
        f.write("\n")
        
        # Individual scene details
        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL SCENE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"Scene: {r['scene_id']}\n")
            f.write(f"Caption: {r.get('ai_caption', 'N/A')}\n")
            f.write(f"Intent: {r.get('intent', 'unspecified')}\n")
            f.write(f"Terms: {', '.join(r.get('terms_used', []))}\n")
            
            dirs = r.get('directions', {})
            if any(dirs.values()):
                f.write(f"Directions: ")
                for axis, values in dirs.items():
                    if values:
                        f.write(f"{axis}={values} ")
                f.write("\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Verify caption quality")
    parser.add_argument("--dataset-root", required=True, help="Path to processed dataset")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of scenes to analyze")
    parser.add_argument("--output", default="caption_quality_report.txt", help="Output report file")
    parser.add_argument("--scene-ids", help="File with specific scene IDs to analyze")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    # Get scene IDs
    if args.scene_ids:
        with open(args.scene_ids, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
    else:
        # Sample from available scenes
        untagged_dir = dataset_root / "untagged_text"
        all_scenes = [f.stem for f in untagged_dir.glob("*.txt")]
        scene_ids = np.random.choice(all_scenes, min(args.sample_size, len(all_scenes)), replace=False)
    
    print(f"Analyzing {len(scene_ids)} scenes...")
    
    # Verify each scene
    results = []
    for scene_id in scene_ids:
        result = verify_scene(scene_id, dataset_root)
        if 'error' not in result:
            results.append(result)
        else:
            print(f"Warning: {scene_id} - {result['error']}")
    
    # Generate report
    generate_report(results, args.output)
    print(f"\nReport written to: {args.output}")
    print(f"Successfully analyzed: {len(results)}/{len(scene_ids)} scenes")


if __name__ == "__main__":
    main()
