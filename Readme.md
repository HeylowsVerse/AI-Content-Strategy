# AI Content Strategy App

This project provides a simple Streamlit application for running marketing experiments on SME customer data.

## Features

- Upload a CSV file containing customer information.
- Automatically determines cluster counts for the `industry` column and for the `product` column inside each industry cluster.
- Performs clustering with K-means and summarises each industry cluster by total amount, average credit score and average tenure, ranking them by total amount.
- Allows you to pick how many of the ranked industry and product clusters to include when configuring the experiment.
- Lets you provide separate keywords for marketing messages and product propositions, specify how many of each to generate, adjust a creative temperature slider and optionally generate suggestions using the **Gemma-2B-IT** model hosted on Hugging Face.
- Randomly assigns customers to message variants.

## Usage

Install the dependencies and run the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload a CSV containing lowercase column names:

- `industry`
- `product`
- `amount`
- `credit_score`
- `tenure`

The provided `sample_customers.csv` includes these columns as well as the
optional fields `cid`, `company_name` and `start_date`. Column names are
automatically normalised to lowercase after upload.

Follow the on-screen instructions to explore clusters, configure your experiment, provide message and product keywords separately and (optionally) generate AI-powered content suggestions.

Gemma model generation requires internet access and may take a long time on first run.
