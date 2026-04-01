import re
import os

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Insert the helper function near the top (e.g. after imports)
helper_func = '''
def render_custom_card(html_content: str):
    """Safely renders HTML UI elements independently of Streamlit's Markdown engine."""
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(html_content, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
'''
content = content.replace("# ═══════════════════════════════════════════════════════════════════════════\n# Theme Constants", helper_func + "# Theme Constants")

# 2. Replace all instances of `st.html(f"""...""")`
content = re.sub(r'st\.html\((\s*f?"""[\s\S]*?""")\)', r'render_custom_card(\1)', content)
content = re.sub(r"st\.html\((\s*f?'''[\s\S]*?''')\)", r"render_custom_card(\1)", content)

# 3. Replace all instances of `st.markdown(f"""...""", unsafe_allow_html=True)`
content = re.sub(r'st\.markdown\((\s*f?"""[\s\S]*?""")\s*,\s*unsafe_allow_html=True\)', r'render_custom_card(\1)', content)
content = re.sub(r"st\.markdown\((\s*f?'''[\s\S]*?''')\s*,\s*unsafe_allow_html=True\)", r"render_custom_card(\1)", content)

# 3.5 Replace all other variations of st.markdown(..., unsafe_allow_html=True)
content = re.sub(r'st\.markdown\((.*?),\s*unsafe_allow_html=True\)', r'render_custom_card(\1)', content)

# Write it back
with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("done")
