from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
import tiktoken 
import networkx as nx
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv
load_dotenv()

class Raptor():
    def __init__(self):
        self.embeddingModel = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        self.model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        self.RANDOM_SEED = 224
    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform global dimensionality reduction on the embeddings using UMAP.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                    If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
    
    def local_cluster_embeddings(
        self, embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - num_neighbors: The number of neighbors to consider for each point.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def get_optimal_clusters(
        self,embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
    ) -> int:
        """
        Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - max_clusters: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.

        Returns:
        - An integer representing the optimal number of clusters found.
        """
        print("Getting oprimal number of cluster using bic...")
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        """
        Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - threshold: The probability threshold for assigning an embedding to a cluster.
        - random_state: Seed for reproducibility.

        Returns:
        - A tuple containing the cluster labels and the number of clusters determined.
        """
        print("Perform GMM clustering...")
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        """
        Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
        using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for UMAP reduction.
        - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

        Returns:
        - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
        """
        print("Perform clustering...")
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]
        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = self.GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # print("Embeddings: ",len(embeddings))
        # print(f"Global clusters: {global_clusters}", n_global_clusters)
        # for i in range(n_global_clusters):
        #     print("ForEach",  np.array([i in gc for gc in global_clusters]))

        # Iterate through each global cluster to perform local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters

    def embed(self,texts):
        """
        Generate embeddings for a list of text documents.

        This function assumes the existence of an `embd` object with a method `embed_documents`
        that takes a list of texts and returns their embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be embedded.

        Returns:
        - numpy.ndarray: An array of embeddings for the given text documents.
        """
        text_embeddings = self.embeddingModel.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np


    def embed_cluster_texts(self,texts):
        """
        Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

        This function combines embedding generation and clustering into a single step. It assumes the existence
        of a previously defined `perform_clustering` function that performs clustering on the embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be processed.

        Returns:
        - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
        """
        text_embeddings = self.embeddingModel.embed_documents(texts)  # Generate embeddings
        text_embeddings_np = np.array(text_embeddings)

        cluster_labels = self.perform_clustering(
            text_embeddings_np, 10, 0.1
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = texts  # Store original texts
        df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df
    
    def fmt_txt(self,df: pd.DataFrame) -> str:
        """
        Formats the text documents in a DataFrame into a single string.

        Parameters:
        - df: DataFrame containing the 'text' column with text documents to format.

        Returns:
        - A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize_texts(
    self, texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
        clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
        the content within each cluster.

        Parameters:
        - texts: A list of text documents to be processed.
        - level: An integer parameter that could define the depth or detail of processing.

        Returns:
        - Tuple containing two DataFrames:
        1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
        2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
            and the cluster identifiers.
        """

        # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
        df_clusters = self.embed_cluster_texts(texts)

        # Prepare to expand the DataFrame for easier manipulation of clusters
        expanded_list = []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        # Summarization
        template = """Here is a sub-set of LangChain Expression Langauge doc. 
            LangChain Expression Langauge provides a way to compose chain in LangChain.Give a detailed summary of the documentation provided.
        
            Documentation:
            {context}
            """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()

        # Format text within each cluster for summarization
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            summaries.append(chain.invoke({"context": formatted_txt}))

        # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary
    
    def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1, n_levels: int = 3
        ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embeds, clusters, and summarizes texts up to a specified level or until
        the number of unique clusters becomes 1, storing the results at each level.

        Parameters:
        - texts: List[str], texts to be processed.
        - level: int, current recursion level (starts at 1).
        - n_levels: int, maximum depth of recursion.

        Returns:
        - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
        levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
        """
        results = {}  # Dictionary to store results at each level

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )

            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)

        return results
    
    def flatten_summaries(self, reptor_results, store_data: List[str]) -> List[str]:
        """
        Flatten summaries from multiple levels of recursion into a single list of strings.

        Returns:
        - List[str]: A list of flattened summaries.
        """
        for level in sorted(reptor_results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = reptor_results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            store_data.extend(summaries)
        return summaries
    
    # HELPER FUNCTIONS
    def num_of_tokens_for_doc(self,doc: str, encoder: str) -> int:
        """
        Calculate the number of tokens in a document.

        Parameters:
        - doc: str, the document for which to calculate the number of tokens.
        - encoder: str, the encoder used to tokenize the document.
        Returns:
        - int, the number of tokens in the document.
        """
        encoder = tiktoken.encoding_for_model(encoder)
        num_tokens = len(encoder.encode(doc))
        # print('num_tokens: ', num_tokens)
        return num_tokens
    


if __name__ == "__main__":
    phrases = [
    "The quick brown fox jumps over the lazy dog",
    "The cats are in the kitchen",
    "Dogs are in the garden",
    "The quick brown fox jumps over the lazy dog",
    "The chicken is in the living room",
    "Lorem ipsum dolor sit amet",
    "Sed ut perspiciatis unde omnis iste natus error",
    "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit",
    "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet",
    "Consectetur adipiscing elit",
    "Nulla facilisi",
    "Suspendisse potenti",
    "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae",
    "Quisque auctor velit sit amet urna malesuada",
    "Cras in mauris ornare, eleifend justo ut, sagittis quam",
    "Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas",
    "Fusce vel justo eget ligula semper consequat",
    "Integer vitae felis sed velit rutrum suscipit",
    "Proin scelerisque ligula non justo consequat, a scelerisque lorem tempor",
    "Nam ac tortor sit amet odio consectetur rhoncus",
    "Aliquam erat volutpat",
    "Etiam vel lectus vehicula, sagittis magna ut, ullamcorper magna",
    "Vivamus dictum lectus eget tempor fermentum",
    "Ut luctus purus non tellus accumsan tincidunt",
    "In luctus turpis vel tincidunt bibendum",
    "Praesent nec eros ut elit vestibulum tincidunt",
    "Curabitur eget mauris at tellus rhoncus consequat",
    "Vestibulum euismod augue a semper laoreet",
    "Morbi nec justo lacinia, ullamcorper ipsum ac, cursus leo",
    "Quisque nec libero eu elit congue elementum",
    "Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas",
    "Fusce sit amet tortor id quam volutpat fringilla",
    "Vestibulum nec ipsum ac enim ultricies bibendum",
    "Nam id nulla quis velit pretium venenatis",
    "In dictum lacus non ante volutpat posuere",
    "Sed id lorem vel mauris suscipit vehicula",
    "Vivamus ultrices risus nec nisl blandit dapibus",
    "Donec scelerisque urna eget magna bibendum ultricies",
    "Nullam nec ligula sed nulla dignissim bibendum",
    "Vestibulum auctor odio vitae fringilla varius"
]


    num_of_tokens = Raptor().num_of_tokens_for_doc("The quick brown fox jumps over the lazy dog", "gpt-4")
    raptoring= Raptor().recursive_embed_cluster_summarize(phrases)
  
            
    with open("raptor_results2.txt", "a") as f:
        for level, (df_clusters, df_summary) in raptoring.items():
            f.write(f"Level {level}\n")
            f.write(f"Clusters:\n{df_clusters}\n")
            f.write(f"Summaries:\n{df_summary}\n")
            f.write("\n")
    summaries= Raptor().flatten_summaries(raptoring, phrases)
    with open("raptor_summaries2.txt", "a") as f:
        for summary in summaries:
            f.write(f"{summary}\n")
            f.write("\n")
