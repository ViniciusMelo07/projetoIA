"""
Análise exploratória inicial do projeto de IA - Core Clothing
Disciplina: Inteligência Artificial

Objetivo:
- Ler um dataset CSV exportado do e-commerce
- Fazer limpeza básica
- Gerar métricas iniciais
- Preparar dados para futura recomendação de produtos

Esperado no CSV (exemplo):
user_id,product_id,product_name,category,price_cents,quantity,event_type,created_at
1,10,Regata Preta,Regatas,7400,1,view,2026-03-01 10:20:00
1,10,Regata Preta,Regatas,7400,1,purchase,2026-03-03 14:10:00
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("core_clothing_events.csv")

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\\n"
            "Exporte um CSV do seu sistema com colunas como: "
            "user_id, product_id, product_name, category, price_cents, quantity, event_type, created_at"
        )
    df = pd.read_csv(path)
    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Padronização mínima
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    if "price_cents" in df.columns:
        df["price_cents"] = pd.to_numeric(df["price_cents"], errors="coerce").fillna(0)

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()

    for col in ["product_name", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

def print_basic_summary(df: pd.DataFrame) -> None:
    print("\\n=== VISÃO GERAL ===")
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")

    if "user_id" in df.columns:
        print(f"Usuários únicos: {df['user_id'].nunique()}")

    if "product_id" in df.columns:
        print(f"Produtos únicos: {df['product_id'].nunique()}")

    if "category" in df.columns:
        print("\\nCategorias:")
        print(df["category"].value_counts(dropna=False).head(10))

    if "event_type" in df.columns:
        print("\\nEventos:")
        print(df["event_type"].value_counts(dropna=False))

def top_products_by_purchase(df: pd.DataFrame) -> pd.DataFrame:
    if not {"event_type", "product_name"}.issubset(df.columns):
        return pd.DataFrame()

    purchases = df[df["event_type"] == "purchase"].copy()
    if purchases.empty:
        return pd.DataFrame()

    if "quantity" not in purchases.columns:
        purchases["quantity"] = 1

    top = (
        purchases.groupby("product_name", as_index=False)["quantity"]
        .sum()
        .sort_values("quantity", ascending=False)
        .head(10)
    )
    return top

def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    required = {"event_type", "category"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    temp = df.copy()
    temp["interaction"] = 1

    grouped = (
        temp.pivot_table(
            index="category",
            columns="event_type",
            values="interaction",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    if "purchase" in grouped.columns and "view" in grouped.columns:
        grouped["conversion_proxy"] = grouped["purchase"] / grouped["view"].replace(0, 1)

    return grouped.sort_values(
        "purchase" if "purchase" in grouped.columns else grouped.columns[-1],
        ascending=False
    )

def generate_chart(top_df: pd.DataFrame) -> None:
    if top_df.empty:
        print("\\nSem dados suficientes para gerar gráfico de produtos comprados.")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(top_df["product_name"], top_df["quantity"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Quantidade comprada")
    plt.xlabel("Produto")
    plt.title("Top 10 produtos por compras")
    plt.tight_layout()
    plt.savefig("top_produtos_core_clothing.png", dpi=200)
    print("\\nGráfico salvo em: top_produtos_core_clothing.png")

def build_user_profile(df: pd.DataFrame, user_id) -> dict:
    if "user_id" not in df.columns:
        return {}

    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        return {}

    profile = {}

    if "category" in user_df.columns:
        profile["categorias_mais_vistas"] = (
            user_df[user_df.get("event_type", "") == "view"]["category"]
            .value_counts()
            .head(3)
            .index
            .tolist()
            if "event_type" in user_df.columns
            else user_df["category"].value_counts().head(3).index.tolist()
        )

    if "product_name" in user_df.columns:
        profile["produtos_relacionados"] = user_df["product_name"].value_counts().head(5).index.tolist()

    return profile

def main() -> None:
    df = load_data(DATA_PATH)
    df = prepare_data(df)

    print_basic_summary(df)

    top = top_products_by_purchase(df)
    if not top.empty:
        print("\\n=== TOP PRODUTOS COMPRADOS ===")
        print(top.to_string(index=False))

    cat_perf = category_performance(df)
    if not cat_perf.empty:
        print("\\n=== DESEMPENHO POR CATEGORIA ===")
        print(cat_perf.to_string(index=False))

    generate_chart(top)

    if "user_id" in df.columns:
        first_user = df["user_id"].dropna().iloc[0]
        print("\\n=== EXEMPLO DE PERFIL DE USUÁRIO ===")
        print(build_user_profile(df, first_user))

if __name__ == "__main__":
    main()
