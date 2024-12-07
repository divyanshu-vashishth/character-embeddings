import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from hindi_embeddings import HindiCharacterEmbedding, get_hindi_dataset

def visualize_embeddings(embedding_model, chars_to_plot):
    """Generate t-SNE visualization of character embeddings"""
    embeddings = []
    valid_chars = []
    
    for char in chars_to_plot:
        try:
            char_embedding = embedding_model.get_character_embedding(char)
            embeddings.append(char_embedding.reshape(-1))
            valid_chars.append(char)
        except ValueError:
            continue
    
    if not embeddings:
        return None
        
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Add labels
    for i, char in enumerate(valid_chars):
        plt.annotate(char, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('t-SNE visualization of Hindi Character Embeddings')
    return plt.gcf()

def generate_model_statistics(embedding_model):
    """Generate basic statistics about the embedding model"""
    stats = {
        'vocabulary_size': len(embedding_model.char_to_idx),
        'embedding_dimension': embedding_model.embedding_dim,
        'unique_characters': sorted(list(embedding_model.char_to_idx.keys()))
    }
    return stats

def generate_report(output_path='report.txt'):
    # Get dataset and train model
    hindi_texts = get_hindi_dataset()
    embedding_model = HindiCharacterEmbedding(embedding_dim=100)
    embedding_model.fit(hindi_texts)
    
    # Generate report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Hindi Character Embeddings - Model Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model statistics
        stats = generate_model_statistics(embedding_model)
        f.write("Model Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Vocabulary Size: {stats['vocabulary_size']}\n")
        f.write(f"Embedding Dimension: {stats['embedding_dimension']}\n")
        f.write(f"Number of Unique Characters: {len(stats['unique_characters'])}\n\n")
        
        # Sample embeddings
        f.write("Sample Character Embeddings:\n")
        f.write("-" * 20 + "\n")
        sample_chars = ["न", "म", "स्", "त", "े"]
        for char in sample_chars:
            try:
                embedding = embedding_model.get_character_embedding(char)
                f.write(f"Character '{char}' embedding shape: {embedding.shape}\n")
            except ValueError as e:
                f.write(f"Error processing character '{char}': {str(e)}\n")
        
        # Generate visualization
        sample_chars = stats['unique_characters'][:50]  # First 50 characters for visualization
        plt.figure(figsize=(12, 8))
        fig = visualize_embeddings(embedding_model, sample_chars)
        if fig:
            fig.savefig('embeddings_visualization.png')
            f.write("\nVisualization saved as 'embeddings_visualization.png'\n")

if __name__ == "__main__":
    generate_report()
