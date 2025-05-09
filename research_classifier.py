import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set page title and configuration
st.set_page_config(page_title="Research Bio Classifier", layout="wide")

# Header
st.title("Research Bio Classification Tool")
st.markdown("Paste a faculty research bio to classify their primary research area")

# Define text area for input
text_area = st.text_area("Paste research bio here:", height=200)


@st.cache_resource
def load_model():
    """Load the sentence transformer model - cached to avoid reloading"""
    return SentenceTransformer('all-MiniLM-L6-v2')


# Load model
model = load_model()

# Define your categories
categories = [
    "Computer and information sciences",
    "Atmospheric science and meteorology",
    "Geological and earth sciences",
    "ocean sciences and marine sciences",
    "Agricultural sciences",
    "Biological and biomedical sciences",
    "Health sciences",
    "Natural resources and conservation",
    "Mathematics and statistics",
    "Astronomy and astrophysics",
    "Chemistry",
    "Materials science",
    "Physics",
    "Psychology",
    "Economics",
    "Political science and government",
    "Sociology, demography, and population studies",
    "Sociology",
    "Aerospace, aeronautical, and astronautical engineering",
    "Bioengineering and biomedical engineering",
    "Chemical engineering",
    "Civil engineering",
    "Electrical, electronic, and communications engineering",
    "Industrial and manufacturing engineering",
    "Mechanical engineering",
    "Metallurgical and materials engineering",
    "Business management and business administration",
    "Education",
    "Humanities",
    "Law",
    "Social work",
    "Visual and performing arts",
]


# Function to classify faculty research based on semantic similarity
def classify_text(text, top_n=2):
    """Classify text and return top n results"""
    if not text.strip():
        return []

    # Generate embeddings for each category
    category_embeddings = model.encode(categories)

    # Generate embedding for the research text
    text_embedding = model.encode([text])[0]

    # Calculate similarity to each category
    similarities = cosine_similarity([text_embedding], category_embeddings)[0]

    # Apply corrections for specific cases
    # Example rule for robotics + NLP
    if ("robot" in text.lower() and
            ("natural language" in text.lower() or "nlp" in text.lower()) and
            categories[similarities.argmax()] == "Psychology"):
        # Find the index of Computer and information sciences
        cs_idx = categories.index("Computer and information sciences")
        # Boost its similarity score
        similarities[cs_idx] = max(0.95, similarities[cs_idx])

    # Add more correction rules as needed

    # Get indices of top n categories
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return top categories and their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'category': categories[idx],
            'confidence': similarities[idx],
            'confidence_pct': f"{similarities[idx]:.1%}"
        })

    return results


# Create the main interface
if st.button("Classify Research"):
    if text_area.strip():
        with st.spinner("Analyzing research bio..."):
            # Get classifications
            results = classify_text(text_area)

            if results:
                # Display results in a nice format
                st.subheader("Classification Results:")

                # Primary match with more details
                st.markdown(f"### Primary Match: {results[0]['category']}")
                st.progress(float(results[0]['confidence']))
                st.markdown(f"Confidence: {results[0]['confidence_pct']}")

                # Secondary matches if available
                if len(results) > 1:
                    st.markdown("### Secondary Matches:")
                    for result in results[1:]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"{result['category']}")
                            st.progress(float(result['confidence']))
                        with col2:
                            st.markdown(f"Confidence: {result['confidence_pct']}")
            else:
                st.error("Unable to classify. Please provide more text.")
    else:
        st.warning("Please enter some text to classify.")

# Add some explanation text
st.markdown("---")
st.markdown("""
### How it works
This tool uses sentence embeddings to analyze research bios and match them with the most relevant research categories.
The model looks at the semantic meaning of the text, not just keywords.
""")

# Add sample text button for easy testing
if st.button("Load Sample Text"):
    sample_text = """My research focuses on combining robotics, machine learning, and natural language processing
    to help robots understand human language and follow instructions. I develop algorithms for semantic understanding
    that enable robots to interpret context and perform complex tasks based on verbal commands."""
    st.session_state.sample_text = sample_text
    st.experimental_rerun()

# Use the session state to populate the text area with the sample
if 'sample_text' in st.session_state:
    text_area = st.text_area("Paste research bio here:", st.session_state.sample_text, height=200, key="bio_text")
    # Clear the session state
    del st.session_state.sample_text