import gradio as gr
from collections import defaultdict, deque

# Define COAL Graph
coal_graph = {
   #enter your coal topics in graph form here
}

# Reverse the graph
reverse_graph = defaultdict(list)
for parent, children in coal_graph.items():
    for child in children:
        reverse_graph[child].append(parent)

# Assign importance from the bottom
def assign_importance_from_bottom(reverse_graph, starting_node="Multimodule Programs"):
    importance = defaultdict(int)
    visited = set()
    queue = deque([(starting_node, 100)])

    while queue:
        node, score = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        importance[node] = score

        for parent in reverse_graph[node]:
            queue.append((parent, score - 1))

    for node in coal_graph:
        if node not in importance:
            importance[node] = 1

    return dict(sorted(importance.items(), key=lambda x: -x[1]))

# Study plan logic
def study_plan(user_input):
    try:
        # First check for the special Arabic response case
        if any(time_word in user_input.lower() for time_word in ["m", "min", "mins", "minute", "minutes"]):
            return "رَبِّ زِدْنِي عِلْمًا"
            
        clean_input = user_input.lower().replace(" ", "")
        
        # Handle combined formats like "2h30m"
        if "h" in clean_input and "m" in clean_input:
            hours_part = clean_input.split("h")[0]
            mins_part = clean_input.split("h")[1].split("m")[0]
            hours = float(hours_part) + float(mins_part)/60
            
        # Handle hour formats
        elif any(time_word in clean_input for time_word in ["h", "hour", "hours"]):
            num = float(''.join([c for c in clean_input if c.isdigit() or c == "."]))
            hours = num
            
        # Handle minute formats (excluding the special Arabic case already handled above)
        elif any(time_word in clean_input for time_word in ["m", "min", "mins", "minute", "minutes"]):
            num = float(''.join([c for c in clean_input if c.isdigit() or c == "."]))
            hours = num / 60
            
        # Handle original simple format ("4 hours")
        else:
            # Try original simple parsing
            try:
                hours = float(user_input.replace("hours", "").strip())
            except:
                return "❌ Please specify time units (e.g., '2h', '1.5 hours' or '30m')"

        num_topics = int(2 + hours)
        importance_scores = assign_importance_from_bottom(reverse_graph)
        top_topics = list(importance_scores.items())[:num_topics]

        result = f"\U0001F4DA Recommended Topics for {hours:.1f} hours of study:\n\n"
        for i, (topic, _) in enumerate(top_topics, 1):
            result += f"{i}. {topic}\n"
        return result
        
    except Exception as e:
        return f"❌ Invalid input. Please enter like '4h', '1.5 hours' or '90m'. Error: {str(e)}"
# This function returns a Gradio column UI to be used inside the main app


def get_coal_study_planner():
    with gr.Column() as planner_ui:
        gr.Markdown("""<h2>ZENTRA - COAL Study Planner 🧠</h2><h3 style='color:gray;'>One-Night before Exam</h3><h4 style='color:gray;'>Prioritized by Conceptual Depth</h4>""")
        with gr.Row():
            time_input = gr.Textbox(
                label="Enter Time (e.g., '4 hours' or '6.5 hours' or '35 minutes')", 
                placeholder="Enter available study time"
            )
            planner_btn = gr.Button("🧠 Generate Study Plan")
        study_output = gr.Textbox(label="ZENTRA's Study Plan", lines=12)
        
        planner_btn.click(fn=study_plan, inputs=time_input, outputs=study_output)
        time_input.submit(fn=study_plan, inputs=time_input, outputs=study_output)
        
    return planner_ui