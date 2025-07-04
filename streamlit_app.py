import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from random import choice

st.set_page_config(page_title="AI Content Strategy App")

st.title("AI Content Strategy App")


def optimal_clusters(values, max_clusters=8):
    X = values.reshape(-1, 1)
    best_k, best_score = 2, -1
    for k in range(2, min(max_clusters, len(values)) + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


uploaded = st.file_uploader("Upload customer CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.lower()
    required = {"industry", "product", "amount", "credit_score", "tenure"}
    if not required.issubset(df.columns):
        st.error(
            "CSV must contain columns: industry, product, amount, credit_score, tenure"
        )
        st.stop()

    ind_enc = LabelEncoder()
    df["industry_code"] = ind_enc.fit_transform(df["industry"].astype(str))

    ind_k = optimal_clusters(df["industry_code"].values)
    st.write(f"Detected {ind_k} industry clusters")

    ind_kmeans = KMeans(n_clusters=ind_k, n_init="auto", random_state=42)
    df["industry_cluster"] = ind_kmeans.fit_predict(df[["industry_code"]])

    df["product_cluster"] = -1
    prod_cluster_counts = {}

    for i in sorted(df["industry_cluster"].unique()):
        subset = df[df["industry_cluster"] == i]
        prod_enc = LabelEncoder()
        codes = prod_enc.fit_transform(subset["product"].astype(str))
        n_unique = len(np.unique(codes))
        if n_unique <= 1:
            prod_k = 1
        else:
            prod_k = optimal_clusters(codes)
        km = KMeans(n_clusters=prod_k, n_init="auto", random_state=42)
        df.loc[subset.index, "product_cluster"] = km.fit_predict(codes.reshape(-1, 1))
        prod_cluster_counts[i] = prod_k

    st.subheader("Industry Cluster Summary")
    summary = (
        df.groupby("industry_cluster")
        .agg(
            industry=("industry", lambda s: ", ".join(sorted(s.unique()))),
            total_amount=("amount", "sum"),
            avg_credit_score=("credit_score", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .sort_values("total_amount", ascending=False)
    )
    st.dataframe(summary)

    sel_ind = st.number_input(
        "How many ranked industry clusters to include?",
        min_value=1,
        max_value=len(summary),
        value=min(3, len(summary)),
    )
    ind_to_use = summary.head(sel_ind).index.tolist()

    selected_prod = defaultdict(int)
    for i in ind_to_use:
        prod_df = df[df["industry_cluster"] == i]
        prod_summary = (
            prod_df.groupby("product_cluster")
            .agg(
                industry=("industry", lambda s: ", ".join(sorted(s.unique()))),
                product=("product", lambda s: ", ".join(sorted(s.unique()))),
                total_amount=("amount", "sum"),
                avg_credit_score=("credit_score", "mean"),
                avg_tenure=("tenure", "mean"),
            )
            .sort_values("total_amount", ascending=False)
        )
        max_prod = prod_cluster_counts[i]
        n_prod = st.number_input(
            f"Industry cluster {i}: how many top product clusters to include?",
            min_value=1,
            max_value=max_prod,
            value=max_prod,
        )
        selected_prod[i] = n_prod
        st.dataframe(prod_summary)

    msg_keywords = st.text_input(
        "Keywords or context for marketing messages",
        key="msg_kw",
    )
    prod_keywords = st.text_input(
        "Keywords or context for product propositions",
        key="prod_kw",
    )
    msg_kw_list = [k.strip() for k in re.split(r"[,\n]+", msg_keywords) if k.strip()]
    prod_kw_list = [k.strip() for k in re.split(r"[,\n]+", prod_keywords) if k.strip()]
    num_message_variants = st.number_input(
        "Number of message variants per industry cluster",
        min_value=1,
        max_value=10,
        value=3,
    )
    num_product_props = st.number_input(
        "Number of product propositions per product cluster",
        min_value=1,
        max_value=10,
        value=3,
    )
    temperature = st.slider(
        "Creative temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    use_llm = st.checkbox(
        "Generate suggestions with Gemma-2B-IT (requires internet and may be slow)",
        value=False,
    )

    if st.button("Configure Experiment"):
        selected_variants = []
        for i in ind_to_use:
            prod_df = df[df["industry_cluster"] == i]
            prod_summary = (
                prod_df.groupby("product_cluster")
                .agg(
                    industry=("industry", lambda s: ", ".join(sorted(s.unique()))),
                    product=("product", lambda s: ", ".join(sorted(s.unique()))),
                    total_amount=("amount", "sum"),
                    avg_credit_score=("credit_score", "mean"),
                    avg_tenure=("tenure", "mean"),
                )
                .sort_values("total_amount", ascending=False)
            )
            prods = prod_summary.head(selected_prod[i]).index.tolist()
            for p in prods:
                selected_variants.append((i, p))

        def generate_text(prompt):
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "google/gemma-2b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
            )
            out = pipe(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
            )
            return out[0]["generated_text"]

        messages = []
        if use_llm and (msg_kw_list or prod_kw_list):
            ind_summary_text = summary.loc[ind_to_use].to_markdown()
            prod_summary_texts = []
            for i in ind_to_use:
                prod_df = df[df["industry_cluster"] == i]
                prod_summary = (
                    prod_df.groupby("product_cluster")
                    .agg(
                        product=("product", lambda s: ", ".join(sorted(s.unique()))),
                        total_amount=("amount", "sum"),
                    )
                    .sort_values("total_amount", ascending=False)
                )
                prod_summary_texts.append(
                    f"Industry {i}:\n" + prod_summary.head(selected_prod[i]).to_markdown()
                )
            prod_text = "\n\n".join(prod_summary_texts)
            prompt = (
                f"You are a marketing strategist. Based on the following cluster summaries, propose {num_message_variants} short marketing message variants for each industry cluster using the keywords: {', '.join(msg_kw_list)}. "
                f"Also propose {num_product_props} product propositions for each product cluster using the keywords: {', '.join(prod_kw_list)}.\n\n"
                f"Industry clusters:\n{ind_summary_text}\n\nProduct clusters:\n{prod_text}"
            )
            try:
                generated = generate_text(prompt)
                messages = [line.strip() for line in generated.split("\n") if line.strip()]
            except Exception as e:
                messages = [f"LLM generation failed: {e}"]

        assignments = []
        for _, row in df.iterrows():
            variant = choice(selected_variants)
            assignments.append(f"I{variant[0]}-P{variant[1]}")
        df["variant"] = assignments

        st.subheader("Variant assignments")
        st.dataframe(df[["industry", "product", "variant"]])

        if messages:
            st.subheader("AI Suggestions")
            for m in messages:
                st.write("-", m)
