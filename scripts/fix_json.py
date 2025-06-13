#!/usr/bin/env python3
import json
import sys

def fix_json(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            content = f.read()

        # Extract the main structure without detailed results
        try:
            # Find where detailed_results starts
            detailed_results_start = content.find('"detailed_results":')
            if detailed_results_start == -1:
                print("Could not find detailed_results in the JSON")
                return False

            # Create a simplified JSON without detailed results
            simplified_content = content[:detailed_results_start] + '"detailed_results": []}'

            # Validate the simplified content
            json.loads(simplified_content)

            with open(output_file, 'w') as f:
                f.write(simplified_content)

            print(f"Fixed JSON saved to {output_file}")
            return True
        except json.JSONDecodeError as e:
            print(f"Could not fix JSON: {e}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_json.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if fix_json(input_file, output_file):
        # Try to load and print summary
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)

            print(f"Accuracy: {data.get('overall_accuracy', 0):.4f}")
            print(f"Correct: {data.get('correct_count', 0)}/{data.get('total_count', 0)}")
            print(f"Avg Latency: {data.get('avg_latency_ms', 0):.2f} ms")
            print(f"P95 Latency: {data.get('p95_latency_ms', 0):.2f} ms")
            print(f"P99 Latency: {data.get('p99_latency_ms', 0):.2f} ms")
            print(f"Images per second: {data.get('images_per_second', 0):.2f}")
        except Exception as e:
            print(f"Error loading fixed JSON: {e}")
    else:
        sys.exit(1)
