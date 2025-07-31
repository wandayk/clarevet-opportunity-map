from dash import Dash, html, dcc
from tkinter import filedialog, Tk, ttk
import tkinter as tk
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State, ALL, MATCH
import webbrowser
from threading import Timer
import locale
import json
locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')

# Importação condicional do dash_bootstrap_components
try:
    import dash_bootstrap_components as dbc
    has_bootstrap = True
except ImportError:
    has_bootstrap = False


def create_sales_dashboard(data):
    # Verificar se temos o bootstrap disponível
    if has_bootstrap:
        app = Dash(__name__, 
                suppress_callback_exceptions=True, 
                external_stylesheets=[dbc.themes.DARKLY])
    else:
        app = Dash(__name__, 
                suppress_callback_exceptions=True)

    # Add this after data conversion, before creating the layout
    # Group data by date for the time series chart
    data_grouped = data.groupby(data['Data Faturamento'].dt.strftime('%Y-%m')).agg({
        'Vda Vl Líquido': 'sum'
    }).reset_index()
    
    # Convert grouped data's date back to datetime for proper sorting
    data_grouped['Data Faturamento'] = pd.to_datetime(data_grouped['Data Faturamento'], format='%Y-%m')
    
    last_12_months = data_grouped.nlargest(12, 'Data Faturamento')
    last_12_months = last_12_months.sort_values('Data Faturamento')
    # Sort data chronologically
    data_grouped = data_grouped.sort_values('Data Faturamento')

    # ...rest of your existing code...
    # Convert date column
    data['Data Faturamento'] = pd.to_datetime(data['Data Faturamento'])
    
    vendedor_nome = data['Vendedor Nome'].iloc[0]

    # Calculate averages based on total customers
    last_month = data['Data Faturamento'].max().strftime('%m/%Y')
    last_month_data = data[data['Data Faturamento'].dt.strftime('%m/%Y') == last_month]
    last_month_revenue = last_month_data['Vda Vl Líquido'].sum()
    total_clientes = len(data['Nome'].unique())

    monthly_revenue = data.groupby(data['Data Faturamento'].dt.strftime('%Y-%m'))['Vda Vl Líquido'].sum()
    media_cliente = monthly_revenue.mean() / total_clientes

    # Add this after creating last_12_months DataFrame
    # Calculate 12-month moving average
    moving_avg = last_12_months['Vda Vl Líquido'].rolling(window=12, min_periods=1).mean()

    # After calculating moving_avg, get the last month's value
    last_month_moving_avg = moving_avg.iloc[-1]  # Gets the last value

    # Add after calculating moving_avg
    def analyze_trend(moving_avg_values):
        # Calculate percentage change over 12 months
        total_growth = ((moving_avg_values.iloc[-1] - moving_avg_values.iloc[0]) / moving_avg_values.iloc[0]) * 100
        
        # Calculate standard deviation to measure stability
        stability = moving_avg_values.pct_change().std()
        
        # Count positive months
        positive_months = (moving_avg_values.pct_change() > 0).sum()
        
        if total_growth >= 10 and positive_months >= 6:
            return {
                'status': 'CRESCIMENTO CONSISTENTE',
                'color': '#2ecc71',
                'message': f'Crescimento sólido de {total_growth:+.1f}% nos últimos 12 meses, com evolução positiva em {positive_months} meses.',
                'growth': total_growth
            }
        elif total_growth > 0:
            return {
                'status': 'CRESCIMENTO MODERADO',
                'color': '#f1c40f',
                'message': f'Crescimento de {total_growth:+.1f}% nos últimos 12 meses, com evolução positiva em {positive_months} meses. Potencial para melhorias.',
                'growth': total_growth
            }
        else:
            return {
                'status': 'ATENÇÃO',
                'color': '#e74c3c',
                'message': f'Queda de {total_growth:.1f}% nos últimos 12 meses. Necessário revisar estratégias comerciais.',
                'growth': total_growth
            }

    trend_analysis = analyze_trend(moving_avg)

    def process_client_data(data):
        # Criar coluna ano-mês para agrupamento e ordenação
        data = data.copy()
        
        # Get the last 12 months of data for display
        latest_date = data['Data Faturamento'].max()
        twelve_months_ago = latest_date - pd.DateOffset(months=11)
        display_data = data[data['Data Faturamento'] >= twelve_months_ago]
        
        # Create month column for grouping
        data['ano_mes'] = data['Data Faturamento'].dt.strftime('%Y-%m-01')
        data['ano_mes'] = pd.to_datetime(data['ano_mes'])
        display_data['ano_mes'] = display_data['Data Faturamento'].dt.strftime('%Y-%m-01')
        display_data['ano_mes'] = pd.to_datetime(display_data['ano_mes'])
        
        # Get unique months for the last 12 months
        unique_months = sorted(display_data['ano_mes'].unique())
        
        # Preparar dados mensais
        monthly_data = []
        all_clients_set = set()  # Conjunto para rastrear todos os clientes únicos
        
        for month in unique_months:
            # Dados do mês atual
            month_mask = data['ano_mes'] == month
            current_month = data[month_mask]
            
            # Dados históricos até o mês atual
            historical_mask = data['ano_mes'] <= month
            historical_data = data[historical_mask]
            
            # Calcular métricas
            monthly_clients = len(current_month['Nome'].unique())
            historical_clients = len(historical_data['Nome'].unique())
            
            monthly_data.append({
                'Data Faturamento': month,  # Usar a data já em formato datetime
                'Total Clientes': monthly_clients,
                'Total Histórico': historical_clients
            })
        
        # Convert to DataFrame
        monthly_clients = pd.DataFrame(monthly_data)
        
        # Calculate absolute growth (difference in number of clients)
        monthly_clients['Crescimento'] = monthly_clients['Total Histórico'].diff()
        
        # Fill first value with 0 instead of NaN
        monthly_clients['Crescimento'] = monthly_clients['Crescimento'].fillna(0).astype(int)
        
        monthly_clients['Crescimento Histórico'] = monthly_clients['Total Histórico'].pct_change() * 100
        
        # Calculate percentage growth from first to last month
        first_month_total = monthly_clients['Total Histórico'].iloc[0]
        last_month_total = monthly_clients['Total Histórico'].iloc[-1]
        total_growth = ((last_month_total - first_month_total) / first_month_total) * 100
        
        return monthly_clients, total_growth

    # In create_sales_dashboard, update how you get the data:
    client_data, client_growth = process_client_data(data)

    last_month_clients = client_data['Total Clientes'].iloc[-1]
    # Average per customer (last month)
    last_month_avg = last_month_revenue / total_clientes

    def process_product_data(data):
        # Get the last 12 months of data and filter for VENDA NORMAL
        latest_date = data['Data Faturamento'].max()
        twelve_months_ago = latest_date - pd.DateOffset(months=11)
        display_data = data[
            (data['Data Faturamento'] >= twelve_months_ago) & 
            (data['Tipo Pedido'] == 'VENDA NORMAL')
        ]
        
        display_data = display_data.copy()
        
        # Criar coluna ano_mes para garantir um dado por mês
        display_data['ano_mes'] = display_data['Data Faturamento'].dt.strftime('%Y-%m-01')
        display_data['ano_mes'] = pd.to_datetime(display_data['ano_mes'])
        
        def process_product_name(name):
            if 'ML' in name.upper() or 'MG' in name.upper() or 'CX' in name.upper():
                return name.split()[0]
            return name
        
        display_data['Produto_Grupo'] = display_data['Produto'].apply(process_product_name)
        
        # Agregar por mês e produto
        monthly_product_data = display_data.groupby([
            'ano_mes', 'Produto_Grupo'
        ])['Vda Vl Líquido'].sum().reset_index()
        
        # Renomear coluna para manter consistência
        monthly_product_data = monthly_product_data.rename(columns={'ano_mes': 'Data Faturamento'})
        
        # Garantir que todos os produtos tenham dados para todos os meses
        unique_months = sorted(monthly_product_data['Data Faturamento'].unique())
        unique_products = monthly_product_data['Produto_Grupo'].unique()
        
        # Criar grid completo de meses e produtos
        month_product_grid = pd.MultiIndex.from_product(
            [unique_months, unique_products],
            names=['Data Faturamento', 'Produto_Grupo']
        ).to_frame(index=False)
        
        # Merge com os dados existentes
        monthly_product_data = month_product_grid.merge(
            monthly_product_data,
            on=['Data Faturamento', 'Produto_Grupo'],
            how='left'
        ).fillna(0)
        
        # Criar resumo total apenas dos últimos 12 meses
        product_summary = display_data.groupby('Produto_Grupo').agg({
            'Vda Qtde Líquida': 'sum',
            'Vda Vl Líquido': 'sum'
        }).reset_index()
        
        product_summary = product_summary.rename(columns={'Produto_Grupo': 'Produto'})
        product_summary = product_summary.sort_values('Vda Vl Líquido', ascending=False)
        
        total_products = len(product_summary)
        total_quantity = product_summary['Vda Qtde Líquida'].sum()
        total_revenue = product_summary['Vda Vl Líquido'].sum()
        
        # Pegar apenas os top 10 produtos para o gráfico
        top_10_products = product_summary.head(10)['Produto'].tolist()
        monthly_product_data = monthly_product_data[
            monthly_product_data['Produto_Grupo'].isin(top_10_products)
        ]
        
        return product_summary, monthly_product_data, total_products, total_quantity, total_revenue
    # Em create_sales_dashboard, adicione:
    product_summary, monthly_product_data, total_products, total_quantity, total_revenue = process_product_data(data)

    # Primeiro, calcule os percentuais para cada produto
    monthly_totals = monthly_product_data.groupby('Data Faturamento')['Vda Vl Líquido'].sum().reset_index()
    monthly_product_data = monthly_product_data.merge(monthly_totals, on=['Data Faturamento'], suffixes=('', '_total'))
    monthly_product_data['percentual'] = monthly_product_data.apply(
        lambda x: (x['Vda Vl Líquido'] / x['Vda Vl Líquido_total'] * 100) if x['Vda Vl Líquido_total'] > 0 else 0, 
        axis=1
    )

    meses = {
    '01': 'Janeiro', '02': 'Fevereiro', '03': 'Março',
    '04': 'Abril', '05': 'Maio', '06': 'Junho',
    '07': 'Julho', '08': 'Agosto', '09': 'Setembro',
    '10': 'Outubro', '11': 'Novembro', '12': 'Dezembro'
    }

    def analyze_municipalities(data):
        """
        Analisa oportunidades por município usando inferência bayesiana
        """
        import numpy as np
        from scipy import stats
        
        # Copiar dados para não modificar o original
        df = data.copy()
        
        # Garantir que a coluna de data seja datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Data Faturamento']):
            df['Data Faturamento'] = pd.to_datetime(df['Data Faturamento'])
        
        # Data mais recente para calcular o tempo decorrido
        latest_date = df['Data Faturamento'].max()
        
        # Resultados por município
        municipality_results = {}
        
        # Para cada município
        for municipality in df['Localização - Município'].unique():
            # Filtrar dados do município
            mun_data = df[df['Localização - Município'] == municipality]
            
            # Ignorar municípios com poucas transações
            if len(mun_data) < 5:
                continue
                
            # Clientes do município
            customers = mun_data['Nome'].unique()
            
            # Estatísticas gerais
            total_customers = len(customers)
            total_revenue = mun_data['Vda Vl Líquido'].sum()
            active_customers = 0
            
            # Variáveis para análise bayesiana
            customer_cycles = []
            customer_recencies = []
            customer_potentials = []
            product_opportunities = {}
            
            # Analisar cada cliente
            for customer in customers:
                customer_data = mun_data[mun_data['Nome'] == customer]
                
                # Obter datas de compra ordenadas
                purchase_dates = sorted(customer_data['Data Faturamento'].unique())
                
                # Calcular ciclo médio de compra (se houver mais de uma compra)
                if len(purchase_dates) > 1:
                    cycles = [(purchase_dates[i+1] - purchase_dates[i]).days 
                            for i in range(len(purchase_dates)-1)]
                    avg_cycle = sum(cycles) / len(cycles) if cycles else 180  # Default 180 dias
                    max_cycle = max(cycles) if cycles else 180
                
                else:
                    avg_cycle = 180  # Valor padrão para clientes com apenas uma compra
                    max_cycle = 180
                
                # Calcular recência (dias desde última compra)
                last_purchase = purchase_dates[-1]
                recency = (latest_date - last_purchase).days
                
                # Considerar cliente ativo se comprou nos últimos 90 dias
                if recency <= 90:
                    active_customers += 1
                
                # Calcular score de ciclo de compra (proporção entre recência e ciclo médio)
                cycle_ratio = recency / avg_cycle if avg_cycle > 0 else 1
                
                # Análise bayesiana para probabilidade de compra
                # Prior beta baseado no histórico de compras
                prior_alpha = len(purchase_dates) + 1  # Compras + 1
                prior_beta = 2  # Suavização
                
                # Likelihood gamma para tempo até próxima compra
                likelihood = stats.gamma.pdf(recency, a=avg_cycle/30, scale=30)
                
                # Calcular probabilidade bayesiana
                if cycle_ratio < 0.5:
                    # Menos da metade do ciclo: baixa probabilidade
                    purchase_prob = 0.1 + (cycle_ratio * 0.4)
                elif cycle_ratio <= 1.2:
                    # Próximo do ciclo: probabilidade média
                    purchase_prob = 0.3 + (cycle_ratio * 0.3)
                elif cycle_ratio > 2:
                    # Mais do dobro do ciclo: alta probabilidade
                    purchase_prob = min(0.9, 0.6 + (0.1 * (cycle_ratio - 2)))
                else:
                    # Caso intermediário
                    purchase_prob = 0.5 + (0.1 * (cycle_ratio - 1.2))
                
                # Ajustar com likelihood
                purchase_prob = min(0.95, purchase_prob * (1 + likelihood))
                
                # Adicionar às listas para calcular média do município
                customer_cycles.append(avg_cycle)
                customer_recencies.append(recency)
                customer_potentials.append(purchase_prob)
                
                # Analisar produtos por cliente
                for product in customer_data['Produto'].unique():
                    if product not in product_opportunities:
                        product_opportunities[product] = {
                            'total_buyers': 0,
                            'potential_buyers': 0,
                            'opportunity_score': 0,
                            'avg_value': 0
                        }
                    
                    # Filtrar compras deste produto
                    product_data = customer_data[customer_data['Produto'] == product]
                    product_opportunities[product]['total_buyers'] += 1
                    
                    # Valor médio de compra
                    avg_value = product_data['Vda Vl Líquido'].mean()
                    product_opportunities[product]['avg_value'] += avg_value
                    
                    # Se há alta probabilidade de compra
                    if purchase_prob >= 0.4:
                        product_opportunities[product]['potential_buyers'] += 1
                        product_opportunities[product]['opportunity_score'] += purchase_prob * avg_value / 1000
            
            # Calcular métricas do município
            if customer_potentials:
                # Score médio de oportunidade
                avg_potential = sum(customer_potentials) / len(customer_potentials)
                # Ciclo médio de compra
                avg_municipality_cycle = sum(customer_cycles) / len(customer_cycles)
                # Recência média
                avg_municipality_recency = sum(customer_recencies) / len(customer_recencies)
                # Ratio médio (recência/ciclo)
                avg_cycle_ratio = avg_municipality_recency / avg_municipality_cycle if avg_municipality_cycle > 0 else 1
                
                # Top produtos com maior oportunidade
                sorted_products = sorted(
                    [(k, v) for k, v in product_opportunities.items() if v['total_buyers'] >= 3],
                    key=lambda x: x[1]['opportunity_score'],
                    reverse=True
                )
                
                top_products = [
                    {
                        'name': p[0],
                        'potential_buyers': p[1]['potential_buyers'],
                        'opportunity_score': p[1]['opportunity_score'],
                        'avg_value': p[1]['avg_value'] / p[1]['total_buyers'] if p[1]['total_buyers'] > 0 else 0
                    }
                    for p in sorted_products[:3]
                ]
                
                # Adicionar aos resultados
                municipality_results[municipality] = {
                    'total_customers': total_customers,
                    'active_customers': active_customers,
                    'avg_potential': avg_potential,
                    'avg_cycle_days': avg_municipality_cycle,
                    'avg_recency_days': avg_municipality_recency,
                    'cycle_ratio': avg_cycle_ratio,
                    'total_revenue': total_revenue,
                    'opportunity_score': avg_potential * (1 + (active_customers / max(total_customers, 1)) * 0.5),
                    'top_products': top_products
                }
        
        return municipality_results

    def create_rj_heatmap(data):
        """
        Cria um mapa de calor de oportunidades para municípios do Rio de Janeiro
        """
        import json
        import requests
        import numpy as np
        
        # Analisar municípios
        try:
            municipality_analysis = analyze_municipalities(data)
        except Exception as e:
            print(f"Erro na análise de municípios: {str(e)}")
            municipality_analysis = {}
        
        # Tentar carregar GeoJSON dos municípios do RJ
        try:
            # URL para o GeoJSON dos municípios do Rio de Janeiro
            municipalities_url = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-33-mun.json"
            response = requests.get(municipalities_url, timeout=5)
            rj_municipalities = response.json()
            
            # Contorno do estado do RJ
            rj_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/main/public/data/brazil-states.geojson"
            response_state = requests.get(rj_url, timeout=5)
            all_states = response_state.json()
            
            # Filtrar apenas o Rio de Janeiro (contorno do estado)
            rj_state = {
                "type": "FeatureCollection",
                "features": [f for f in all_states["features"] if f["properties"].get("name") == "Rio de Janeiro"]
            }
        except Exception as e:
            print(f"Erro ao carregar GeoJSON: {str(e)}")
            # GeoJSON simplificado em caso de falha
            rj_municipalities = None
            rj_state = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Rio de Janeiro"},
                        "geometry": {"type": "Polygon", "coordinates": [[
                            [-44.8, -21.4], [-41.0, -21.4], 
                            [-41.0, -23.4], [-44.8, -23.4]
                        ]]}
                    }
                ]
            }
        
        # Criar figura
        fig = go.Figure()
        
        # Coordenadas dos municípios do Rio de Janeiro que devem ser definidas ANTES da função create_rj_heatmap
        municipality_coordinates = {
            "ANGRA DOS REIS": {"lat": -23.0067, "lon": -44.3183},
            "APERIBÉ": {"lat": -21.6253, "lon": -42.1017},
            "ARARUAMA": {"lat": -22.8729, "lon": -42.3432},
            "AREAL": {"lat": -22.2283, "lon": -43.1118},
            "ARMAÇÃO DOS BÚZIOS": {"lat": -22.7469, "lon": -41.8846},
            "ARRAIAL DO CABO": {"lat": -22.9662, "lon": -42.0278},
            "BARRA DO PIRAÍ": {"lat": -22.4715, "lon": -43.8269},
            "BARRA MANSA": {"lat": -22.5481, "lon": -44.1752},
            "BELFORD ROXO": {"lat": -22.7631, "lon": -43.3994},
            "BOM JARDIM": {"lat": -22.1544, "lon": -42.4125},
            "BOM JESUS DO ITABAPOANA": {"lat": -21.1355, "lon": -41.6822},
            "CABO FRIO": {"lat": -22.8894, "lon": -42.0286},
            "CACHOEIRAS DE MACACU": {"lat": -22.4658, "lon": -42.6539},
            "CAMBUCI": {"lat": -21.5686, "lon": -41.9189},
            "CAMPOS DOS GOYTACAZES": {"lat": -21.7545, "lon": -41.3244},
            "CANTAGALO": {"lat": -21.9797, "lon": -42.3664},
            "CARAPEBUS": {"lat": -22.1875, "lon": -41.6631},
            "CARDOSO MOREIRA": {"lat": -21.4846, "lon": -41.6131},
            "CARMO": {"lat": -21.9308, "lon": -42.6203},
            "CASIMIRO DE ABREU": {"lat": -22.4812, "lon": -42.2042},
            "COMENDADOR LEVY GASPARIAN": {"lat": -22.0381, "lon": -43.2066},
            "CONCEIÇÃO DE MACABU": {"lat": -22.0833, "lon": -41.8719},
            "CORDEIRO": {"lat": -22.0258, "lon": -42.3644},
            "DUAS BARRAS": {"lat": -22.0536, "lon": -42.5172},
            "DUQUE DE CAXIAS": {"lat": -22.7869, "lon": -43.3116},
            "ENGENHEIRO PAULO DE FRONTIN": {"lat": -22.5398, "lon": -43.6827},
            "GUAPIMIRIM": {"lat": -22.5347, "lon": -42.9895},
            "IGUABA GRANDE": {"lat": -22.8494, "lon": -42.2297},
            "ITABORAÍ": {"lat": -22.7565, "lon": -42.9614},
            "ITAGUAÍ": {"lat": -22.8535, "lon": -43.7785},
            "ITALVA": {"lat": -21.4259, "lon": -41.7014},
            "ITAOCARA": {"lat": -21.6748, "lon": -42.0758},
            "ITAPERUNA": {"lat": -21.1997, "lon": -41.8799},
            "ITATIAIA": {"lat": -22.4973, "lon": -44.5675},
            "JAPERI": {"lat": -22.6432, "lon": -43.6601},
            "LAJE DO MURIAÉ": {"lat": -21.2091, "lon": -42.1271},
            "MACAÉ": {"lat": -22.3768, "lon": -41.7848},
            "MACUCO": {"lat": -21.9813, "lon": -42.2531},
            "MAGÉ": {"lat": -22.6512, "lon": -43.0418},
            "MANGARATIBA": {"lat": -22.9597, "lon": -44.0406},
            "MARICÁ": {"lat": -22.9195, "lon": -42.8195},
            "MENDES": {"lat": -22.5245, "lon": -43.7312},
            "MESQUITA": {"lat": -22.7830, "lon": -43.4300},
            "MIGUEL PEREIRA": {"lat": -22.4572, "lon": -43.4803},
            "MIRACEMA": {"lat": -21.4148, "lon": -42.1956},
            "NATIVIDADE": {"lat": -21.0389, "lon": -41.9697},
            "NILÓPOLIS": {"lat": -22.8057, "lon": -43.4229},
            "NITERÓI": {"lat": -22.8832, "lon": -43.1033},
            "NOVA FRIBURGO": {"lat": -22.2819, "lon": -42.5310},
            "NOVA IGUAÇU": {"lat": -22.7592, "lon": -43.4511},
            "PARACAMBI": {"lat": -22.6078, "lon": -43.7108},
            "PARAÍBA DO SUL": {"lat": -22.1585, "lon": -43.2846},
            "PARATY": {"lat": -23.2178, "lon": -44.7131},
            "PATY DO ALFERES": {"lat": -22.4309, "lon": -43.4285},
            "PETRÓPOLIS": {"lat": -22.5112, "lon": -43.1779},
            "PINHEIRAL": {"lat": -22.5173, "lon": -44.0022},
            "PIRAÍ": {"lat": -22.6215, "lon": -43.8880},
            "PORCIÚNCULA": {"lat": -20.9632, "lon": -42.0465},
            "PORTO REAL": {"lat": -22.4175, "lon": -44.2952},
            "QUATIS": {"lat": -22.4041, "lon": -44.2597},
            "QUEIMADOS": {"lat": -22.7208, "lon": -43.5524},
            "QUISSAMÃ": {"lat": -22.1031, "lon": -41.4693},
            "RESENDE": {"lat": -22.4705, "lon": -44.4509},
            "RIO BONITO": {"lat": -22.7179, "lon": -42.6276},
            "RIO CLARO": {"lat": -22.7204, "lon": -44.1419},
            "RIO DAS FLORES": {"lat": -22.1692, "lon": -43.5856},
            "RIO DAS OSTRAS": {"lat": -22.5174, "lon": -41.9475},
            "RIO DE JANEIRO": {"lat": -22.9068, "lon": -43.1729},
            "SANTA MARIA MADALENA": {"lat": -21.9547, "lon": -42.0098},
            "SANTO ANTÔNIO DE PÁDUA": {"lat": -21.5393, "lon": -42.1832},
            "SÃO FIDÉLIS": {"lat": -21.6413, "lon": -41.7536},
            "SÃO FRANCISCO DE ITABAPOANA": {"lat": -21.4702, "lon": -41.1091},
            "SÃO GONÇALO": {"lat": -22.8269, "lon": -43.0539},
            "SÃO JOÃO DA BARRA": {"lat": -21.6384, "lon": -41.0514},
            "SÃO JOÃO DE MERITI": {"lat": -22.7997, "lon": -43.3713},
            "SÃO JOSÉ DE UBÁ": {"lat": -21.3661, "lon": -41.9511},
            "SÃO JOSÉ DO VALE DO RIO PRETO": {"lat": -22.1525, "lon": -42.9327},
            "SÃO PEDRO DA ALDEIA": {"lat": -22.8429, "lon": -42.1026},
            "SÃO SEBASTIÃO DO ALTO": {"lat": -21.9578, "lon": -42.1328},
            "SAPUCAIA": {"lat": -21.9952, "lon": -42.9142},
            "SAQUAREMA": {"lat": -22.9292, "lon": -42.5099},
            "SEROPÉDICA": {"lat": -22.7526, "lon": -43.7155},
            "SILVA JARDIM": {"lat": -22.6574, "lon": -42.3961},
            "SUMIDOURO": {"lat": -22.0485, "lon": -42.6761},
            "TANGUÁ": {"lat": -22.7423, "lon": -42.7202},
            "TERESÓPOLIS": {"lat": -22.4167, "lon": -42.9833},
            "TRAJANO DE MORAES": {"lat": -22.0638, "lon": -42.0643},
            "TRÊS RIOS": {"lat": -22.1119, "lon": -43.2092},
            "VALENÇA": {"lat": -22.2449, "lon": -43.7037},
            "VARRE-SAI": {"lat": -20.9327, "lon": -41.8701},
            "VASSOURAS": {"lat": -22.4059, "lon": -43.6685},
            "VOLTA REDONDA": {"lat": -22.5202, "lon": -44.0950}
        }
        # Adicionar o contorno do estado como base
        fig.add_trace(go.Choropleth(
            geojson=rj_state,
            locations=[f["properties"]["name"] for f in rj_state["features"]],
            z=[1],  # Valor fixo
            colorscale=[[0, 'rgba(30,30,30,0.8)'], [1, 'rgba(30,30,30,0.8)']],
            marker_line_color='white',
            marker_line_width=1.5,
            showscale=False,
            name='',
            featureidkey="properties.name"
        ))
        
        # Preparar dados de municípios para o mapa
        municipality_markers = []
        
        if municipality_analysis:
            # Normalizar os scores para o tamanho dos marcadores
            max_score = max([m['opportunity_score'] for m in municipality_analysis.values()]) if municipality_analysis else 1
            
            for municipality, metrics in municipality_analysis.items():
                # Normalizar o nome do município para comparação
                municipality_upper = municipality.upper().replace("-", " ").strip()
                
                # Procurar o município nas coordenadas
                found = False
                for mun_name, coords in municipality_coordinates.items():
                    if municipality_upper == mun_name or municipality_upper in mun_name:
                        # Preparar recomendações de produtos
                        product_recommendations = ""
                        if metrics['top_products']:
                            for idx, product in enumerate(metrics['top_products']):
                                product_recommendations += f"<br>• {product['name']} ({product['potential_buyers']} potenciais)"
                                if idx >= 2:  # Limitar a 3 produtos
                                    break
                        
                        # Calcular tamanho normalizado do marcador
                        size = (metrics['opportunity_score'] / max_score * 20) + 5  # Mínimo 5, máximo 25
                        
                        # Definir cor com base no ciclo de compra vs recência
                        if metrics['cycle_ratio'] > 1.5:
                            color = 'rgba(231, 76, 60, 0.8)'  # Vermelho (alta urgência)
                        elif metrics['cycle_ratio'] > 0.8:
                            color = 'rgba(241, 196, 15, 0.8)'  # Amarelo (média urgência)
                        else:
                            color = 'rgba(46, 204, 113, 0.8)'  # Verde (baixa urgência)
                        
                        municipality_markers.append({
                            "name": municipality,
                            "lat": coords["lat"],
                            "lon": coords["lon"],
                            "size": min(size, 25),  # Limitar tamanho máximo
                            "color": color,
                            "customers": metrics["total_customers"],
                            "active": metrics["active_customers"],
                            "potential": metrics["avg_potential"] * 100,  # Converter para porcentagem
                            "cycle": metrics["avg_cycle_days"],
                            "recency": metrics["avg_recency_days"],
                            "recommendations": product_recommendations,
                            "revenue": metrics["total_revenue"]
                        })
                        found = True
                        break
        
        # Adicionar marcadores de município com dados
        if municipality_markers:
            fig.add_trace(go.Scattergeo(
                lon=[m["lon"] for m in municipality_markers],
                lat=[m["lat"] for m in municipality_markers],
                text=[m["name"] for m in municipality_markers],
                mode='markers+text',
                marker=dict(
                    size=[m["size"] for m in municipality_markers],
                    color=[m["color"] for m in municipality_markers],
                    line_color='white',
                    line_width=1,
                    sizemode='diameter'
                ),
                textposition="middle right",
                textfont=dict(color="white", size=11),
                customdata=[[
                    m["customers"],
                    m["active"],
                    m["potential"],
                    m["cycle"],
                    m["recency"],
                    m["revenue"],
                    m["recommendations"]
                ] for m in municipality_markers],
                hovertemplate="<b>%{text}</b><br>Clientes: %{customdata[0]}<br>Ativos: %{customdata[1]} (%{customdata[2]:.1f}% potencial)<br>Faturamento: R$ %{customdata[5]:,.2f}<extra></extra>",
                name=''
            ))
        
        # Adicionar linhas de municípios como uma camada separada
        if rj_municipalities:
            try:
                # Extrair coordenadas dos municípios para traçar linhas
                for feature in rj_municipalities["features"]:
                    if feature["geometry"]["type"] == "Polygon":
                        for coord_list in feature["geometry"]["coordinates"]:
                            lons = [coord[0] for coord_list in feature["geometry"]["coordinates"] for coord in coord_list]
                            lats = [coord[1] for coord_list in feature["geometry"]["coordinates"] for coord in coord_list]
                            fig.add_trace(go.Scattergeo(
                                lon=lons,
                                lat=lats,
                                mode="lines",
                                line=dict(width=0.5, color="rgba(255, 255, 255, 0.5)"),
                                hoverinfo="skip",
                                showlegend=False
                            ))
                    elif feature["geometry"]["type"] == "MultiPolygon":
                        for multi_poly in feature["geometry"]["coordinates"]:
                            for coord_list in multi_poly:
                                lons = [coord[0] for coord_list in multi_poly for coord in coord_list]
                                lats = [coord[1] for coord_list in multi_poly for coord in coord_list]
                                fig.add_trace(go.Scattergeo(
                                    lon=lons,
                                    lat=lats,
                                    mode="lines",
                                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.5)"),
                                    hoverinfo="skip",
                                    showlegend=False
                                ))
            except Exception as e:
                print(f"Erro ao traçar linhas de municípios: {str(e)}")
        
       # Substituir a parte de update_geos na função create_rj_heatmap:

        # Ajustar layout do mapa com zoom automático baseado nos municípios com dados
        if municipality_markers:
            # Extrair coordenadas de municípios com dados
            lats = [m["lat"] for m in municipality_markers]
            lons = [m["lon"] for m in municipality_markers]
            
            # Calcular limites com margem
            min_lat = min(lats) - 0.1
            max_lat = max(lats) + 0.1
            min_lon = max(lons) - 0.8  # Margem horizontal maior
            max_lon = max(lons) + 0.8  # Margem horizontal maior
            
            # Garantir que a proporção seja adequada para visualização (mais larga que alta)
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            
            # Forçar uma proporção de ~2:1 (largura:altura)
            desired_ratio = 3.0  # Largura é 2x a altura
            current_ratio = lon_range / lat_range
            
            if current_ratio < desired_ratio:
                # Aumentar a largura para atingir a proporção desejada
                additional_lon = (desired_ratio * lat_range - lon_range) / 3
                min_lon -= additional_lon
                max_lon += additional_lon
            
        else:
            # Valores padrão para todo o estado do RJ se não houver municípios com dados
            # Usando proporção de ~2:1 (largura:altura)
            min_lat = -23.8
            max_lat = -20.7
            min_lon = -45.5  # Mais amplo horizontalmente
            max_lon = -40.3  # Mais amplo horizontalmente

        # Ajustar layout do mapa
        fig.update_geos(
            projection_type="mercator",
            center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2},  # Centro baseado nos municípios com dados
            scope="south america",
            # Limites dinâmicos baseados nos municípios com dados
            lataxis_range=[min_lat, max_lat],
            lonaxis_range=[min_lon, max_lon],
            visible=False,
            resolution=50,
            showcoastlines=False,
            coastlinecolor="white",
            showland=True,
            landcolor="rgba(50, 50, 50, 0.2)",
            showocean=True,
            oceancolor="rgba(0, 0, 0, 0)",
            showlakes=False,
            showrivers=False,
            bgcolor='rgba(0,0,0,0)'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=60, t=0, b=0),
            showlegend=False,
            height=420,
            width=None,  # Permite que a largura seja responsiva
            hoverlabel=dict(
                bgcolor="rgb(40, 40, 40)",
                bordercolor="white",
                font=dict(family="Roboto", size=12, color="white"),
                align="left",
                namelength=-1
            ),
        )
        
        return fig

    # Format month string
    mes_num = last_month.split('/')[0]
    mes_extenso = meses[mes_num]
    ano = last_month.split('/')[1]

    # Calculate year-over-year growth
    def calculate_growth(data):
        # Group by year-month
        monthly = data.groupby(data['Data Faturamento'].dt.strftime('%Y-%m'))['Vda Vl Líquido'].sum()
        # Calculate growth comparing same month previous year
        growth = monthly.pct_change(periods=12) * 100
        # Convert to dictionary and handle NaN values
        return {k: v if pd.notnull(v) else None for k, v in growth.items()}

    # Get growth rates
    growth_rates = calculate_growth(data)

    def analyze_client_opportunity(data, client_name=None):
        """
        Analisa oportunidades de venda para um cliente específico usando inferência bayesiana
        """
        import numpy as np
        from scipy import stats
        
        if client_name is None:
            # Selecionar um cliente aleatório que tenha pelo menos 5 compras
            clients_with_history = data.groupby('Nome').filter(lambda x: len(x) >= 5)['Nome'].unique()
            if len(clients_with_history) > 0:
                client_name = np.random.choice(clients_with_history)
            else:
                # Fallback para qualquer cliente
                client_name = np.random.choice(data['Nome'].unique())
        
        # Filtrar dados do cliente
        client_data = data[data['Nome'] == client_name].copy()
        client_data = client_data.sort_values('Data Faturamento')
        
        # Dados gerais do cliente
        total_purchases = len(client_data)
        total_spent = client_data['Vda Vl Líquido'].sum()
        last_purchase = client_data['Data Faturamento'].max()
        days_since_last = (pd.Timestamp.today() - last_purchase).days
        
        # Análise de produtos comprados
        product_analysis = []
        products = client_data['Produto'].unique()
        
        # Encontrar o último valor de venda de cada produto para qualquer cliente
        last_product_prices = {}
        for product in products:
            # Obter o último preço registrado para este produto em todo o dataset
            product_data = data[data['Produto'] == product].sort_values('Data Faturamento', ascending=False)
            if not product_data.empty:
                last_unit_price = product_data.iloc[0]['Vda Vl Líquido'] / product_data.iloc[0]['Vda Qtde Líquida'] if product_data.iloc[0]['Vda Qtde Líquida'] > 0 else 0
                last_product_prices[product] = last_unit_price
        
        for product in products:
            product_purchases = client_data[client_data['Produto'] == product].copy()
            product_purchases = product_purchases.sort_values('Data Faturamento')
            
            # Calcular gaps entre compras
            if len(product_purchases) > 1:
                purchase_dates = product_purchases['Data Faturamento']
                gaps = [(purchase_dates.iloc[i+1] - purchase_dates.iloc[i]).days 
                    for i in range(len(purchase_dates)-1)]
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                
                # Dias desde a última compra deste produto
                days_since_product = (pd.Timestamp.today() - product_purchases['Data Faturamento'].iloc[-1]).days
                
                # Análise Bayesiana para probabilidade de compra
                # Prior: distribuição beta baseada no histórico de compras
                prior_alpha = len(gaps) + 1  # Número de compras + 1
                prior_beta = 2  # Valor inicial para suavização
                
                # Likelihood: tempo desde a última compra vs. tempo médio de recompra
                # Usamos uma distribuição gamma para modelar o tempo até a próxima compra
                likelihood = stats.gamma.pdf(days_since_product, a=avg_gap/30, scale=30)
                
                # Ajustar pela razão entre dias passados e gap médio
                gap_ratio = days_since_product / avg_gap if avg_gap > 0 else 0
                
                # Probabilidade bayesiana de compra
                if gap_ratio < 0.5:
                    # Se passou menos da metade do tempo médio, probabilidade baixa
                    purchase_prob = 0.1 + (gap_ratio * 0.4)
                elif gap_ratio <= 1.2:
                    # Se próximo do tempo médio, probabilidade média
                    purchase_prob = 0.3 + (gap_ratio * 0.3)
                elif gap_ratio > 2:
                    # Se passou muito tempo, alta probabilidade
                    purchase_prob = min(0.9, 0.6 + (0.1 * (gap_ratio - 2)))
                else:
                    # Caso intermediário
                    purchase_prob = 0.5 + (0.1 * (gap_ratio - 1.2))
                
                # Ajustar com likelihood
                purchase_prob = min(0.95, purchase_prob * (1 + likelihood))
                
                # Quantidade provável de compra (média histórica)
                avg_quantity = product_purchases['Vda Qtde Líquida'].mean()
                total_quantity = product_purchases['Vda Qtde Líquida'].sum()
                
                # Estimar compra futura com base no padrão
                last_cycle_qty = product_purchases['Vda Qtde Líquida'].iloc[-1]
                predicted_qty = max(1, avg_quantity * (1 + 0.1 * np.random.randn()))
                
                if last_cycle_qty < avg_quantity:
                    # Se última compra foi abaixo da média, provável que próxima seja maior
                    predicted_qty = avg_quantity * 1.2
                
                # Arredondar para número inteiro
                predicted_qty = round(predicted_qty)
                
                # Obter o último valor de venda deste produto (para qualquer cliente)
                ultimo_valor = last_product_prices.get(product, 0)
                
                product_analysis.append({
                    'produto': product,
                    'total_compras': len(product_purchases),
                    'qtd_total': total_quantity,
                    'qtd_media': avg_quantity,
                    'ultima_compra': product_purchases['Data Faturamento'].iloc[-1],
                    'gap_medio': avg_gap,
                    'gap_maximo': max_gap,
                    'dias_desde_ultima': days_since_product,
                    'prob_compra': purchase_prob,
                    'qtd_prevista': predicted_qty,
                    'valor_medio': product_purchases['Vda Vl Líquido'].mean() / product_purchases['Vda Qtde Líquida'].mean() if product_purchases['Vda Qtde Líquida'].mean() > 0 else 0,
                    'ultimo_valor': ultimo_valor,  # Adicionando o último valor de venda
                    'urgencia': days_since_product / avg_gap if avg_gap > 0 else 0
                })
        
        # Ordenar produtos por probabilidade de compra
        product_analysis.sort(key=lambda x: x['prob_compra'], reverse=True)
        
        return {
            'nome': client_name,
            'total_compras': total_purchases,
            'valor_total': total_spent,
            'ultima_compra': last_purchase,
            'dias_desde_ultima': days_since_last,
            'produtos': product_analysis
        }

    # No final do layout, após o mapa, adicione:
    client_analysis = analyze_client_opportunity(data)

    def create_client_card(client_analysis, idx=None):
        # Ordenar produtos por probabilidade de compra
        sorted_products = sorted(client_analysis['produtos'], key=lambda x: x['prob_compra'], reverse=True)
        
        # Calcular o faturamento potencial (produtos com prob >= 0.5)
        potential_revenue = 0
        
        for prod in sorted_products:
            if prod['prob_compra'] >= 0.5:  # Considerar apenas produtos com probabilidade >= 50%
                # Usar o último valor de venda em vez da média
                item_value = prod['ultimo_valor'] * prod['qtd_prevista']
                potential_revenue += item_value
        
        # Definir a cor de urgência baseada no produto com maior probabilidade
        top_product = sorted_products[0] if sorted_products else None
        
        if top_product and top_product['prob_compra'] >= 0.7:
            urgency_color = '#e74c3c'  # Vermelho para alta probabilidade
            urgency_text = 'ALTA OPORTUNIDADE'
            indicator_emoji = '🔴'
        elif top_product and top_product['prob_compra'] >= 0.5:
            urgency_color = '#f1c40f'  # Amarelo para média probabilidade
            urgency_text = 'MÉDIA OPORTUNIDADE'
            indicator_emoji = '🟡'
        else:
            urgency_color = '#2ecc71'  # Verde para baixa probabilidade
            urgency_text = 'BAIXA OPORTUNIDADE'
            indicator_emoji = '🟢'
        
        # Calcular dias desde a última compra para estilo visual
        days_text = f"{client_analysis['dias_desde_ultima']} dias"
        days_color = 'white'
        if client_analysis['dias_desde_ultima'] > 90:
            days_color = '#e74c3c'  # Vermelho se > 90 dias
        elif client_analysis['dias_desde_ultima'] > 45:
            days_color = '#f39c12'  # Laranja se > 45 dias
        
        # Usar index único para o componente
        card_index = str(idx) if idx is not None else client_analysis['nome'].replace(' ', '-')
        
        # Converter o dicionário client_analysis para string JSON para armazenar como data-attribute
        import json
        client_json = json.dumps(client_analysis, default=str)  # default=str para lidar com datas
        
        # Criar o card com atributo data-client em vez de data
        return html.Div([
            # Cabeçalho moderno com indicador de oportunidade
            html.Div([
                html.Div([
                    html.Div(indicator_emoji, style={
                        'font-size': '14px',
                        'line-height': '1',
                        'margin-right': '8px'
                    }),
                    html.H4(client_analysis['nome'], style={
                        'font-weight': '600',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '15px',
                        'font-family': 'roboto',
                        'overflow': 'hidden',
                        'text-overflow': 'ellipsis',
                        'white-space': 'nowrap',
                        'flex-grow': '1'
                    })
                ], style={
                    'display': 'flex', 
                    'align-items': 'center',
                    'width': '100%'
                }),
                
                # Badge de oportunidade
                html.Div(urgency_text, style={
                    'font-weight': '500',
                    'color': 'white',
                    'font-size': '10px',
                    'background': urgency_color,
                    'padding': '3px 8px',
                    'border-radius': '12px',
                    'letter-spacing': '0.5px',
                    'margin-top': '8px',
                    'align-self': 'flex-start',
                    'box-shadow': '0 1px 3px rgba(0,0,0,0.2)'
                })
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'border-bottom': f'1px solid rgba(255,255,255,0.1)',
                'padding-bottom': '12px',
                'margin-bottom': '12px'
            }),
            
            # Conteúdo principal - métricas em layout de grade moderna
            html.Div([
                # Valor total
                html.Div([
                    html.P("Valor Total", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(f"R$ {client_analysis['valor_total']:,.2f}", style={
                        'color': 'white',
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px'}),
                
                # Total de compras
                html.Div([
                    html.P("Compras", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(f"{client_analysis['total_compras']}", style={
                        'color': 'white',
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px', 'border-left': '1px solid rgba(255,255,255,0.08)'}),
                
                # Última compra
                html.Div([
                    html.P("Última Compra", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(days_text, style={
                        'color': days_color,
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px', 'border-left': '1px solid rgba(255,255,255,0.08)'})
            ], style={
                'display': 'grid',
                'grid-template-columns': '1fr 1fr 1fr',
                'margin-bottom': '16px',
                'background-color': 'rgba(255,255,255,0.03)',
                'border-radius': '8px'
            }),
            
            # Faturamento potencial - highlight
            html.Div([
                html.Div([
                    html.Span('POTENCIAL', style={
                        'font-size': '11px',
                        'font-weight': '600',
                        'color': '#ffba3c',
                        'letter-spacing': '0.5px'
                    }),
                    html.H2(f"R$ {potential_revenue:,.2f}", style={
                        'margin': '4px 0 0 0',
                        'font-size': '18px',
                        'font-weight': '700',
                        'color': 'white'
                    })
                ]),
                
                # Seta indicadora
                html.Div('→', style={
                    'color': '#ffba3c',
                    'font-size': '22px',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                })
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',
                'align-items': 'center',
                'background': 'linear-gradient(90deg, rgba(255,186,60,0.1) 0%, rgba(255,186,60,0.03) 100%)',
                'border-left': '3px solid #ffba3c',
                'border-radius': '4px',
                'padding': '10px 16px',
                'margin-top': '8px'
            }),
            
            # Div oculta para armazenar os dados do cliente
            html.Div(id={'type': 'client-data', 'index': card_index}, **{'data-client': client_json}, style={'display': 'none'})
        ], 
        id={'type': 'client-card', 'index': card_index}, 
        n_clicks=0,  # Inicializar contador de cliques
        # Removi o atributo 'data' que estava causando o erro
        style={
            'backgroundColor': 'rgb(35, 35, 35)',
            'borderRadius': '8px',
            'padding': '16px',
            'height': '100%',
            'border': '1px solid rgba(255, 255, 255, 0.05)',
            'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
            'transition': 'transform 0.2s, box-shadow 0.2s',
            'cursor': 'pointer',
            'position': 'relative',
            'overflow': 'hidden',
            'transform': 'translateY(0)',
            '&:hover': {
                'transform': 'translateY(-5px)',
                'boxShadow': '0 8px 15px rgba(0,0,0,0.2)'
            }
        })

    app.layout = html.Div([
        # Add Google Fonts import
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700;800;900&display=swap",
            rel="stylesheet"
        ),
        html.Div([
             html.Img(src='https://static.wixstatic.com/media/bf137f_cf5b6a22899e447b84a4aaa2775799e9~mv2.png/v1/fill/w_253,h_253,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/LOGOCLAREVET.png'
                      ,style={'width':'70px','margin':'0 40px','border-bottom': '2px solid orange'}),
             html.Div([
                html.Div([
                     html.A('FATURAMENTO', href='#faturamento', style={
                        'margin': '0',
                        'font-weight': '600',
                        'color': 'white',
                        'font-size': '13px', 
                        'text-transform': 'full-width', 
                        'cursor': 'pointer',
                        'text-decoration': 'none'  # Remover sublinhado padrão de links
                    }),
                    html.A('CLIENTES', href='#clientes', style={
                        'margin': '0',
                        'font-weight': '600',
                        'color': 'white',
                        'font-size': '13px', 
                        'text-transform': 'full-width', 
                        'cursor': 'pointer',
                        'text-decoration': 'none'
                    }),
                    html.A('PRODUTOS', href='#produtos', style={
                        'margin': '0',
                        'font-weight': '600',
                        'color': 'white',
                        'font-size': '13px', 
                        'text-transform': 'full-width', 
                        'cursor': 'pointer',
                        'text-decoration': 'none'
                    }),
                    html.A('MAPA DE OPORTUNIDADE', href='#oportunidades', style={
                        'margin': '0',
                        'font-weight': '600',
                        'color': 'white',
                        'font-size': '13px', 
                        'text-transform': 'full-width', 
                        'cursor': 'pointer',
                        'text-decoration': 'none'
                    }),
                ], style={'display': 'flex','flex-direction': 'row','align-items': 'end','padding-right': '40px', 'justify-content': 'space-between','gap': '80px'})
             ], style={'font-size': '18px', 'font-weight': '500','color': '#6b6b6b','display': 'flex','align-items': 'center','gap': '10px'}),
        ], style={
            'font-family':'Roboto',
            'display': 'flex', 
            'align-items': 'center',
            'padding': '0 20px',
            'border-bottom': '1px solid #80808059',
            'z-index': '10',
            'gap':'50px', 
            'justify-content':'space-between',
            'backdrop-filter':' blur(20px)',
            'background': '#0000',
            'position': 'fixed',
            'width': '100%',
            'height': '75px'
        }),
        
        # FATURAMENTO
        
        html.Div([

            # Calculate last month's total revenue
        
        # Add div with card showing total revenue
        html.Div([
            html.Div([
                    html.H1('Dashboard', style={'margin': '0','font-weight': '700','color': 'rgb(255, 255, 255)','font-size': '30px'}),
                    html.P('Relatório de vendas', style={'margin': '0','font-weight': '600','color': '#6d6d6d','font-size': '13px'}),
                ], style={'display': 'flex','font-family': 'Roboto','gap': '5px','flex-direction': 'column'}),
            html.Div([
                    html.H1('VENDEDOR', style={
                        'margin': '0px',
                        'font-weight': '600',
                        'color': '#ffba3c',
                        'font-size': '12px',
                        }),
                    html.P(f'{vendedor_nome}', style={
                        'margin': '0px',
                        'font-weight': '500',
                        'color': '#f9f9f9',
                        'font-size': '13px',
                        'background': '#313131',
                        'padding': '10px 30px',
                        'border-radius': '5px',
                        'border': '1px solid #ffba3c57',
                        'line-height': '1',
                        'box-shadow': 'rgba(0, 0, 0, 0.25) 0px 4px 8px -2px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px'
                        }),
                ], style={'display': 'flex','font-family': 'Roboto','align-items': 'end','flex-direction': 'column','gap': '5px'})
            ], style={'display': 'flex','font-family': 'Roboto','align-items': 'center','padding': '0 60px','justify-content': 'space-between', 'padding-top': '60px'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H4('FATURAMENTO',style={
                        
                        'font-weight': '700',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '19px',
                        'font-family': 'roboto'
                        }),
                    html.P('DOS ÚLTIMOS', style={
                       
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109',
                        'margin': '0',
                        'font-size': '12px'
                        }),
                    html.H2('DOZE MESES', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109',
                        'margin': '0',
                        'font-size': '12px'
                        }),
                ], style={
                        'border-bottom': '2px solid orange',
                        'padding-bottom': '10px'
                        }),
                 html.Div([
                    html.H4('MÉDIA MÓVEL DOS ÚLTIMOS DOZE MESES',style={
                        
                        'font-weight': '400',
                        'color': 'rgb(107, 107, 107)',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '13px'
                    }),
                    html.H2(f'R$ {last_month_moving_avg:,.2f}', style={
                        'margin': '0',
                        'text-wrap': 'nowrap',
                        'padding-top': '3px',
                        'font-size': '20px',
                        'color': 'white',
                    }),
                    html.Div([
                        html.H2(trend_analysis['status'], style={
                            'margin': '0',
                            'text-wrap': 'nowrap',
                            'font-size': '12px',
                            'color': trend_analysis['color'],
                            'font-weight': '600',

                        }),
                        html.P(f"{trend_analysis['growth']:+.1f}% em 12 meses", style={
                            'margin': '10px 0',
                            'font-size': '13px',
                            'color': 'white',
                        }),
                        html.P(trend_analysis['message'], style={
                            'margin': '0',
                            'font-size': '13px',
                            'color': 'rgb(107, 107, 107)',
                            'line-height': '1.4'
                        })
                    ],style={
                       
                        'font-weight': '400',
                        'color': 'rgb(107, 107, 107)',
                        'margin-top': '20px',
                        'font-size': '13px',
                        'border': '2px solid #ffa50038',
                        'border-radius': '5px',
                        'padding': '20px'
                    })
                ],style={'margin-top':'20px','display':'flex','flex-direction':'column','gap':'7px'}),
            ], style={
                'borderRadius': '8px',
                'textAlign': 'initial',
                'display': 'flex',
                'flex-direction': 'column',
                'height': 'fit-content',
                'width': '270px'
            }),
            # Time series chart by vendor
            html.Div([
                dcc.Graph(
                    figure=go.Figure()
                    .add_trace(
                        go.Scatter(
                            x=last_12_months['Data Faturamento'],
                            y=last_12_months['Vda Vl Líquido'],
                            fill='tozeroy',
                            fillcolor='rgba(255, 186, 60, 0.1)',
                            line=dict(color='rgba(255, 186, 60, 0.0)'),
                            text=last_12_months.apply(
                                lambda x: f"R$ {x['Vda Vl Líquido']:,.2f}",
                                axis=1
                            ),
                            textposition='bottom center',
                            textfont=dict(
                                size=12,
                                color='white'
                            ),
                            mode='lines',
                            showlegend=False
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=last_12_months['Data Faturamento'],
                            y=last_12_months['Vda Vl Líquido'],
                            line=dict(color='#ffba3c', width=2),
                            mode='lines+markers+text',
                            marker=dict(
                                color='#ffba3c',
                                size=8
                            ),
                            text=[
                                f"<b><span style='font-size:11px;padding-bottom:15px;color: {'#2ecc71' if growth_rates.get(date.strftime('%Y-%m')) and growth_rates.get(date.strftime('%Y-%m'), 0) >= 0 else '#e74c3c'}'>{round(growth_rates.get(date.strftime('%Y-%m'), 0)):+}%</span></b>" 
                                if growth_rates.get(date.strftime('%Y-%m')) is not None 
                                else ""
                                for date in last_12_months['Data Faturamento']
                            ],
                            texttemplate='<br>%{text}',
                            textposition='bottom center',
                            showlegend=False,
                            hovertemplate='%{x|%B/%Y}<br>R$ %{y:,.2f}'  # New hover template
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=last_12_months['Data Faturamento'],
                            y=moving_avg,
                            line=dict(
                                color='rgba(0, 150, 255, 0.8)',  # Blue color with some transparency
                                width=2,
                                dash='dot'  # Make it a dotted line
                            ),
                            name='Média Móvel',
                            mode='lines',
                            showlegend=False,
                            hovertemplate='Média: R$ %{y:,.2f}'
                        )
                    )
                   .update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title='',
                        yaxis_title='',
                        xaxis=dict(
                        tickfont=dict(color='white'),
                        ticktext=last_12_months['Data Faturamento'].dt.strftime('%b/%y').str.capitalize(),  # Formato abreviado pt-br
                        tickangle=0,
                        showgrid=False,
                        tickmode='array',
                        tickvals=last_12_months['Data Faturamento'],
                    ),
                        yaxis=dict(
                            showgrid=False,  # Remove y-axis grid
                            showticklabels=False,
                            tickfont_size=12,
                            tickprefix=15,
                            zeroline=False  # Remove y-axis zero line
                        ),
                                            # ... existing layout settings ...
                        margin=dict(l=0, r=0, t=30, b=40),  # Increased top margin
                        showlegend=False,
                        autosize=True,
                        # Add these new settings
                                               annotations=[
                            dict(
                                x=date,
                                y=value,
                                text=f'<span>R$ {value:,.2f}</span>',
                                showarrow=False,
                                yshift=28,
                                font=dict(
                                    color='white',
                                    size=12,
                                    family='Roboto'
                                ),
                                align='center',
                                bgcolor='rgb(49, 49, 49)',  # Background color
                                bordercolor='rgba(255, 186, 60, 0.34)',  # Border color
                                borderwidth=1,
                                borderpad=3
                            )
                            for date, value in zip(last_12_months['Data Faturamento'], last_12_months['Vda Vl Líquido'])
                        ],
                        uniformtext=dict(
                            mode='hide',  # Hide labels that would overlap
                            minsize=8     # Minimum text size
                        )
                    )
                )
                ], style={'width': '100%', 'height': '100%'})  # Make container fill available space
        ], id='faturamento', style={'display': 'flex', 'gap': '100px', 'justify-content': 'space-between', 'font-family': 'Roboto', 'padding': '0 60px', 'padding-top': '60px', 'scroll-margin-top': '75px', 'min-height': 'calc(100vh - 135px)'}),

       
        
        # CLIENTES

       
        
        # Replace the existing CLIENTES section with this new code
        html.Div([
            html.Div([
                html.Div([
                    html.H4('CLIENTES',style={
                        'font-weight': '700',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '19px',
                        'font-family': 'roboto'
                    }),
                    html.P('HISTÓRICO DOS ÚLTIMOS', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                    html.H2('DOZE MESES', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                ], style={
                    'border-bottom': '2px solid orange',
                    'padding-bottom': '10px'
                }),
                html.Div([
                    html.H4('TOTAL DE CLIENTES', style={
                        'font-weight': '400',
                        'color': 'rgb(107, 107, 107)',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '13px'
                    }),
                    html.H2(f"{client_data['Total Histórico'].iloc[-1]:,.0f}", style={
                        'margin': '0',
                        'text-wrap': 'nowrap',
                        'padding-top': '3px',
                        'font-size': '20px',
                        'color': 'white',
                    }),
                    html.P(f"{client_growth:+.1f}% de crescimento no período", style={
                        'margin': '10px 0',
                        'font-size': '13px',
                        'color': '#2ecc71' if client_growth >= 0 else '#e74c3c',
                    })
                ], style={'margin-top':'20px','display':'flex','flex-direction':'column','gap':'7px'})
            ], style={
                'borderRadius': '8px',
                'textAlign': 'initial',
                'display': 'flex',
                'flex-direction': 'column',
                'height': 'fit-content',
                'width': '270px'
            }),
                                                # Atualiza o gráfico para usar os valores acumulados
                                               html.Div([
                            dcc.Graph(
                                figure=go.Figure()
                                .add_trace(
                                    go.Scatter(
                                        x=client_data['Data Faturamento'],
                                        y=client_data['Total Histórico'],
                                        fill='tozeroy',
                                        fillcolor='rgba(6, 126, 210, 0.1)',
                                        line=dict(color='rgba(255, 186, 60, 0.0)'),
                                        mode='lines',
                                        showlegend=False
                                    )
                                )
                                .add_trace(
                                    go.Scatter(
                                        x=client_data['Data Faturamento'],
                                        y=client_data['Total Clientes'],
                                        fill='tozeroy',
                                        fillcolor='rgba(255, 186, 60, 0.14)',
                                        line=dict(color='rgba(255, 186, 60, 0.0)'),
                                        mode='lines',
                                        showlegend=False
                                    )
                                )
                                .add_trace(
                                    go.Scatter(
                                        x=client_data['Data Faturamento'],
                                        y=client_data['Total Clientes'],
                                        line=dict(color='#ffba3c', width=2),
                                        mode='lines+markers+text',
                                        marker=dict(
                                            color='#ffba3c',
                                            size=8
                                        ),
                                        name='Clientes do Mês',
                                        text=[
                                        f"<b><span>{val:,.0f}</span></b>"
                                            for val in client_data['Total Clientes']
                                        ],
                                        texttemplate='<br>%{text}',
                                        textposition='top center',
                                        textfont=dict(
                                            size=12,
                                            color='white'
                                        ),
                                        showlegend=True,
                                        hovertemplate='%{x|%B/%Y}<br>Clientes do Mês: %{y:,.0f}'
                                    )
                                )
                                
                                .add_trace(
                                    go.Scatter(
                                        x=client_data['Data Faturamento'],
                                        y=client_data['Total Clientes'],
                                        mode='text',
                                        text=[
                                            f"<span>{(monthly/historical)*100:.1f}%</span>" if historical > 0 else "<span>0.0%</span>"
                                            for monthly, historical in zip(client_data['Total Clientes'], client_data['Total Histórico'])
                                        ],
                                        texttemplate='<br>%{text}',
                                        textposition='bottom center',
                                        textfont=dict(
                                            size=10,
                                            color='#c6c5c5'
                                        ),
                                        showlegend=True,
                                    )
                                )

                                .add_trace(
                                    go.Scatter(
                                        x=client_data['Data Faturamento'],
                                        y=client_data['Total Histórico'],
                                        line=dict(
                                            color='rgba(0, 150, 255, 0.8)',
                                            width=2,
                                            dash='dot'
                                        ),
                                        name='Total Histórico',
                                        mode='lines+markers+text',
                                        marker=dict(
                                            color='rgba(0, 150, 255, 0.8)',
                                            size=8
                                        ),
                                        text=[
                                            f"<b><span style='font-size:11px;padding-bottom:15px;color: {'#2ecc71' if val >= 0 else '#e74c3c'}'>{val:+}</span></b>"
                                            if not pd.isna(val) else ""
                                            for val in client_data['Crescimento']
                                        ],
                                        textposition='bottom center',
                                        textfont=dict(
                                            size=12,
                                            color='white'
                                        ),
                                        texttemplate='<br>%{text}',
                                        showlegend=True,
                                        hovertemplate='%{x|%B/%Y}<br>Total Histórico: %{y:,.0f}',
                                        
                                    )
                                )
                                .update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    xaxis_title='',
                                    yaxis_title='',
                                    xaxis=dict(
                                        tickfont=dict(color='white'),
                                        ticktext=client_data['Data Faturamento'].dt.strftime('%b/%y').str.capitalize(),
                                        tickangle=0,
                                        showgrid=False,
                                        tickmode='array',
                                        tickvals=client_data['Data Faturamento'],
                                    ),
                                    yaxis=dict(
                                        showgrid=False,
                                        showticklabels=False,
                                        zeroline=False
                                    ),
                                    margin=dict(l=0, r=0, t=30, b=40),
                                    showlegend=False,
                                    legend=dict(
                                        orientation='h',
                                        yanchor='bottom',
                                        y=1.02,
                                        xanchor='right',
                                        x=1,
                                        font=dict(color='white'),
                                        bgcolor='rgba(0,0,0,0)'
                                    ),
                                    autosize=True,
                                    annotations=[
                                        
                                        dict(
                                            x=date,
                                            y=value,
                                            text=f'<span>{value:,.0f}</span>',
                                            showarrow=False,
                                            yshift=28,
                                            font=dict(
                                                color='white',
                                                size=12,
                                                family='Roboto'
                                            ),
                                            align='center',
                                            bgcolor='rgb(49, 49, 49)',
                                            bordercolor='rgba(0, 150, 255, 0.8)',
                                            borderwidth=1,
                                            borderpad=3
                                        )
                                        for date, value in zip(client_data['Data Faturamento'], client_data['Total Histórico'])
                                        
                                    ]
                                    
                                )
                            )
                        ], style={'width': '100%', 'height': '100%'})
        ], id='clientes', style={'display': 'flex', 'gap': '100px', 'justify-content': 'space-between', 'font-family': 'Roboto', 'padding': '0 60px', 'padding-top': '60px', 'scroll-margin-top': '75px', 'min-height': 'calc(100vh - 135px)'}),
        






         # PRODUTOS

       
        
        # Replace the existing PRODUTOS section with this new code
        html.Div([
            html.Div([
                html.Div([
                    html.H4('PRODUTOS',style={
                        'font-weight': '700',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '19px',
                        'font-family': 'roboto'
                    }),
                    html.P('HISTÓRICO DOS ÚLTIMOS', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                    html.H2('DOZE MESES', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                ], style={
                    'border-bottom': '2px solid orange',
                    'padding-bottom': '10px'
                }),
                html.Div([
                    html.H4('TOP 10 PRODUTOS', style={
                        'font-weight': '700',
                        'color': '#c4c4c4',
                        'margin': '0px 0px 10px 0px',
                        'padding': '0',
                        'font-size': '13px'
                    }),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div('Produto', style={'color': 'rgb(107, 107, 107)', 'text-align': 'left'}),
                                html.Div('Faturamento', style={'color': 'rgb(107, 107, 107)', 'text-align': 'right'})
                            ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '10px', 'margin-bottom': '10px'}),
                        ]),
                        html.Div([
                            html.Div([
                                html.Div(row['Produto'], style={'color': 'white'}),
                
                                html.Div(f"R$ {row['Vda Vl Líquido']:,.2f}", style={'color': 'white', 'text-align': 'right'})
                            ], style={
                                'display': 'flex',
                                'justify-content': 'space-between', 
                                'gap': '10px', 
                                'margin-bottom': '5px',
                                'border': '1px solid #ffa5005c',
                                'border-radius': '3px',
                                'padding': '5px',
                                'font-size': '10px',
                                'background': '#ffa50014'
                            }) 
                            for i, row in product_summary.head(10).iterrows()
                        ])
                    ], style={
                        'width': '100%',
                        'border-spacing': '0 8px',
                        'font-size': '13px'
                    })
                ], style={'margin-top':'20px','display':'flex','flex-direction':'column','gap':'7px'})
            ], style={
                'borderRadius': '8px',
                'textAlign': 'initial',
                'display': 'flex',
                'flex-direction': 'column',
                'height': 'fit-content',
                'width': '270px'
            }),
                                                # Atualiza o gráfico para usar os valores acumulados
            html.Div([
                       # Na seção do gráfico de produtos, substitua o dcc.Graph existente por:
            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Scatter(
                                name=produto,
                                x=monthly_product_data[monthly_product_data['Produto_Grupo'] == produto]['Data Faturamento'],
                                y=monthly_product_data[monthly_product_data['Produto_Grupo'] == produto]['percentual'],
                                stackgroup='one',
                                mode='lines',
                                fill='tonexty',
                                opacity=0.06,
                                line=dict(width=2.5, shape='spline'),  # Linhas suavizadas
                                hovertemplate='<b>%{x|%b/%y}</b><br>' + produto + '<br>R$ %{customdata:,.2f}<br>%{y:.1f}% do total',
                                customdata=monthly_product_data[monthly_product_data['Produto_Grupo'] == produto]['Vda Vl Líquido'],
                                showlegend=False
                            )
                            for produto in product_summary.head(10)['Produto']
                        ]
                    ).add_traces([
                        # Adicionar marcadores destacados nos pontos de dados
                        go.Scatter(
                            name=produto,
                                x=monthly_product_data[monthly_product_data['Produto_Grupo'] == produto]['Data Faturamento'],
                                y=monthly_product_data[monthly_product_data['Produto_Grupo'] == produto]['percentual'],
                            mode='markers',
                            marker=dict(size=5, opacity=0),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                        for produto in product_summary.head(10)['Produto']
                    ]).update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title='',
                        yaxis_title='',
                        xaxis=dict(
                            tickfont=dict(color='white', size=11),
                            ticktext=monthly_product_data['Data Faturamento'].dt.strftime('%b/%y').str.capitalize(),
                            tickangle=0,
                            showgrid=False,
                            tickmode='array',
                            tickvals=monthly_product_data['Data Faturamento'].unique(),
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            range=[0, 100]
                        ),
                        margin=dict(l=0, r=0, t=50, b=20),
                        showlegend=False,
                        autosize=True,
                        height=500,
                        shapes=[
                            # Linha horizontal para 100%
                            dict(
                                type="line",
                                xref="paper", yref="y",
                                x0=0, y0=100, x1=1, y1=100,
                                line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
                            ),
                            # Linhas verticais para cada mês (pontos X)
                            *[dict(
                                type="line",
                                xref="x", yref="paper",
                                x0=date, y0=0, x1=date, y1=1,
                                line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot")
                            ) for date in monthly_product_data['Data Faturamento'].unique()]
                        ]
                    ).add_shape(  # Linha horizontal em 100%
                        type="line",
                        xref="paper", yref="y",
                        x0=0, y0=100, x1=1, y1=100,
                        line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
                    ).add_annotation(  # Label para 100%
                        x=0, y=100,
                        xref="paper", yref="y",
                        text="100%",
                        showarrow=False,
                        xanchor="left",
                        font=dict(color="white", size=10),
                        bgcolor="rgba(30, 30, 30, 0.7)",
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=3
                    )
                )
            ], style={'width': '100%', 'height': '100%'})
        ], style={'width': '100%', 'height': '100%'}),

        ], id='produtos', style={'display': 'flex', 'gap': '100px', 'justify-content': 'space-between', 'font-family': 'Roboto', 'padding': '0 60px', 'padding-top': '60px', 'scroll-margin-top': '75px', 'min-height': 'calc(100vh - 135px)'}),
        
                # Substitua a segunda seção de CLIENTES duplicada pelo mapa de oportunidades do Rio de Janeiro
        html.Div([
            html.Div([
                html.Div([
                    html.H4('MAPA DE OPORTUNIDADES', style={
                        'font-weight': '700',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '19px',
                        'font-family': 'roboto'
                    }),
                    html.P('ANÁLISE GEOGRÁFICA DO', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                    html.H2('RIO DE JANEIRO', style={
                        'font-weight': '400',
                        'color': 'rgb(109, 109, 109)',
                        'margin': '0',
                        'font-size': '12px'
                    }),
                ], style={
                    'border-bottom': '2px solid orange',
                    'padding-bottom': '10px'
                }),
                html.Div([
                    html.H4('POTENCIAL POR REGIÃO', style={
                        'font-weight': '400',
                        'color': 'rgb(107, 107, 107)',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '13px'
                    }),
                    html.P('Mapa de calor representando a densidade de clientes e oportunidades de venda por região', style={
                        'margin': '10px 0',
                        'font-size': '13px',
                        'color': 'white',
                        'line-height': '1.4'
                    }),
                    html.P('Clique nas regiões para ver detalhes', style={
                        'margin-top': '20px',
                        'font-size': '13px',
                        'color': 'rgb(107, 107, 107)',
                    }),
                    html.P('🔴 Alta oportunidade', style={
                        'margin': '0',
                        'font-size': '12px',
                        'color': 'white',
                    }),
                    html.P('🟡 Média oportunidade', style={
                        'margin': '0',
                        'font-size': '12px',
                        'color': 'white',
                    }),
                    html.P('🟢 Baixa oportunidade', style={
                        'margin': '0',
                        'font-size': '12px',
                        'color': 'white',
                    }),
                    html.P('Tamanho do círculo = Potencial de venda', style={
                        'margin': '0',
                        'font-size': '12px',
                        'color': 'white',
                    }),
                ], style={
                    'borderRadius': '8px',
                    'textAlign': 'initial',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'height': 'fit-content',
                    'width': '270px'
                }),
                html.Div([
                    dcc.Graph(
                        id='rj-heatmap',
                        figure=create_rj_heatmap(data),
                        style={'height': '100%'}
                    )
                ], style={'width': '100%', 'height': '100%'})
            ], id='oportunidades', style={'display': 'flex', 'gap': '100px', 'justify-content': 'space-between', 'font-family': 'Roboto', 'padding': '0 60px', 'padding-top': '60px', 'scroll-margin-top': '75px', 'min-height': 'calc(100vh - 75px)'}),

            # Div para mostrar os clientes do município selecionado
            html.Div(id='clientes-municipio-container', children=[
                # Será preenchido dinamicamente quando um município for clicado
            ], style={
                'display': 'none',  # Inicialmente oculto
                'padding': '20px 60px 60px 60px',
                'margin-top': '20px',
                'font-family': 'Roboto',
                'width': '100%',
                'scroll-margin-top': '75px'
            })
        ], style={'overflow': 'auto', 'display': 'flex', 'flex-direction': 'column', 'height': '100%', 'padding-top': '75px'})
    ], style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'display': 'flex', 'flex-direction': 'column', 'background': 'rgb(30, 30, 30)'})

    # Adicione ao final de app.layout 
    app.layout.children.extend([
        dcc.Location(id='url', refresh=False),
        html.Div(id='scroll-trigger', style={'display': 'none'}),
        dcc.Store(id='all-client-data', data={}),  # Adicionar este componente
        
        # Modal para detalhes do cliente
        html.Div([
            html.Div([
                # Conteúdo do modal será preenchido pelo callback
                html.Div(id='modal-content')
            ], style={
                'backgroundColor': 'rgb(40, 40, 40)',
                'padding': '25px',
                'borderRadius': '8px',
                'width': '70%',
                'maxWidth': '900px',
                'maxHeight': '85vh',
                'overflowY': 'auto',
                'position': 'relative',
                'boxShadow': '0 5px 15px rgba(0,0,0,0.5)',
                'zIndex': '1002'
            })
        ], id='client-modal', style={
            'display': 'none',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.7)',
            'zIndex': '1000',
            'justifyContent': 'center',
            'alignItems': 'center',
            'padding': '20px'
        })
    ])
    
    def create_client_card(client_analysis, idx=None):
        # Ordenar produtos por probabilidade de compra
        sorted_products = sorted(client_analysis['produtos'], key=lambda x: x['prob_compra'], reverse=True)
        
        # Calcular o faturamento potencial (produtos com prob >= 0.5)
        potential_revenue = 0
        
        for prod in sorted_products:
            if prod['prob_compra'] >= 0.5:  # Considerar apenas produtos com probabilidade >= 50%
                # Usar o último valor de venda em vez da média
                item_value = prod['ultimo_valor'] * prod['qtd_prevista']
                potential_revenue += item_value
        
        # Definir a cor de urgência baseada no produto com maior probabilidade
        top_product = sorted_products[0] if sorted_products else None
        
        if top_product and top_product['prob_compra'] >= 0.7:
            urgency_color = '#e74c3c'  # Vermelho para alta probabilidade
            urgency_text = 'ALTA OPORTUNIDADE'
            indicator_emoji = '🔴'
        elif top_product and top_product['prob_compra'] >= 0.5:
            urgency_color = '#f1c40f'  # Amarelo para média probabilidade
            urgency_text = 'MÉDIA OPORTUNIDADE'
            indicator_emoji = '🟡'
        else:
            urgency_color = '#2ecc71'  # Verde para baixa probabilidade
            urgency_text = 'BAIXA OPORTUNIDADE'
            indicator_emoji = '🟢'
        
        # Calcular dias desde a última compra para estilo visual
        days_text = f"{client_analysis['dias_desde_ultima']} dias"
        days_color = 'white'
        if client_analysis['dias_desde_ultima'] > 90:
            days_color = '#e74c3c'  # Vermelho se > 90 dias
        elif client_analysis['dias_desde_ultima'] > 45:
            days_color = '#f39c12'  # Laranja se > 45 dias
        
        # Usar index único para o componente
        card_index = str(idx) if idx is not None else client_analysis['nome'].replace(' ', '-')
        
        return html.Div([
            # Cabeçalho moderno com indicador de oportunidade
            html.Div([
                html.Div([
                    html.Div(indicator_emoji, style={
                        'font-size': '14px',
                        'line-height': '1',
                        'margin-right': '8px'
                    }),
                    html.H4(client_analysis['nome'], style={
                        'font-weight': '600',
                        'color': 'white',
                        'margin': '0',
                        'padding': '0',
                        'font-size': '15px',
                        'font-family': 'roboto',
                        'overflow': 'hidden',
                        'text-overflow': 'ellipsis',
                        'white-space': 'nowrap',
                        'flex-grow': '1'
                    })
                ], style={
                    'display': 'flex', 
                    'align-items': 'center',
                    'width': '100%'
                }),
                
                # Badge de oportunidade
                html.Div(urgency_text, style={
                    'font-weight': '500',
                    'color': 'white',
                    'font-size': '10px',
                    'background': urgency_color,
                    'padding': '3px 8px',
                    'border-radius': '12px',
                    'letter-spacing': '0.5px',
                    'margin-top': '8px',
                    'align-self': 'flex-start',
                    'box-shadow': '0 1px 3px rgba(0,0,0,0.2)'
                })
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'border-bottom': f'1px solid rgba(255,255,255,0.1)',
                'padding-bottom': '12px',
                'margin-bottom': '12px'
            }),
            
            # Conteúdo principal - métricas em layout de grade moderna
            html.Div([
                # Valor total
                html.Div([
                    html.P("Valor Total", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(f"R$ {client_analysis['valor_total']:,.2f}", style={
                        'color': 'white',
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px'}),
                
                # Total de compras
                html.Div([
                    html.P("Compras", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(f"{client_analysis['total_compras']}", style={
                        'color': 'white',
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px', 'border-left': '1px solid rgba(255,255,255,0.08)'}),
                
                # Última compra
                html.Div([
                    html.P("Última Compra", style={
                        'color': '#8a8a8a',
                        'margin': '0 0 3px 0',
                        'font-size': '11px',
                        'font-weight': '500'
                    }),
                    html.H3(days_text, style={
                        'color': days_color,
                        'margin': '0',
                        'font-size': '14px',
                        'font-weight': '600'
                    })
                ], style={'padding': '8px', 'border-left': '1px solid rgba(255,255,255,0.08)'})
            ], style={
                'display': 'grid',
                'grid-template-columns': '1fr 1fr 1fr',
                'margin-bottom': '16px',
                'background-color': 'rgba(255,255,255,0.03)',
                'border-radius': '8px'
            }),
            
            # Faturamento potencial - highlight
            html.Div([
                html.Div([
                    html.Span('POTENCIAL', style={
                        'font-size': '11px',
                        'font-weight': '600',
                        'color': '#ffba3c',
                        'letter-spacing': '0.5px'
                    }),
                    html.H2(f"R$ {potential_revenue:,.2f}", style={
                        'margin': '4px 0 0 0',
                        'font-size': '18px',
                        'font-weight': '700',
                        'color': 'white'
                    })
                ]),
                
                # Seta indicadora
                html.Div('→', style={
                    'color': '#ffba3c',
                    'font-size': '22px',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                })
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',
                'align-items': 'center',
                'background': 'linear-gradient(90deg, rgba(255,186,60,0.1) 0%, rgba(255,186,60,0.03) 100%)',
                'border-left': '3px solid #ffba3c',
                'border-radius': '4px',
                'padding': '10px 16px',
                'margin-top': '8px'
            })
        ], id={'type': 'client-card', 'index': card_index}, 
           n_clicks=0,  # Inicializar contador de cliques
           data=client_analysis,  # Armazenar dados completos do cliente
           style={
            'backgroundColor': 'rgb(35, 35, 35)',
            'borderRadius': '8px',
            'padding': '16px',
            'height': '100%',
            'border': '1px solid rgba(255, 255, 255, 0.05)',
            'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
            'transition': 'transform 0.2s, box-shadow 0.2s',
            'cursor': 'pointer',
            'position': 'relative',
            'overflow': 'hidden',
            '&:hover': {
                'transform': 'translateY(-5px)',
                'boxShadow': '0 8px 15px rgba(0,0,0,0.2)'
            }
        })

    def analyze_municipality_clients(data, municipality):
        """
        Analisa todos os clientes de um município específico
        """
        # Filtrar dados do município
        mun_data = data[data['Localização - Município'] == municipality]
        
        # Se não houver dados para este município
        if len(mun_data) == 0:
            return []
        
        # Obter lista de clientes únicos
        clients = mun_data['Nome'].unique()
        
        # Analisar cada cliente
        client_analyses = []
        
        for client in clients:
            # Usar a função analyze_client_opportunity existente
            analysis = analyze_client_opportunity(data, client)
            client_analyses.append(analysis)
        
        # Ordenar por probabilidade de compra (considerando o primeiro produto de cada cliente)
        client_analyses.sort(key=lambda x: max([p['prob_compra'] for p in x['produtos']]) if x['produtos'] else 0, reverse=True)
        
        return client_analyses
    
    @app.callback(
        [Output('clientes-municipio-container', 'children'),
         Output('clientes-municipio-container', 'style'),
         Output('all-client-data', 'data'),
         Output('scroll-trigger', 'children')],
        [Input('rj-heatmap', 'clickData')],
        [State('clientes-municipio-container', 'style')]
    )
    def display_municipality_clients(clickData, current_style):
        if not clickData:
            return [], {'display': 'none'}, {}, False
        
        try:
            # Extrair o nome do município do clickData
            municipality = clickData['points'][0]['text']
            
            # Analisar os clientes do município usando a função definida no escopo
            client_analyses = analyze_municipality_clients(data, municipality)
            
            if not client_analyses:
                return [
                    html.H3(f"Não foram encontrados clientes para {municipality}", 
                           style={'color': 'white', 'text-align': 'center', 'margin-top': '30px'})
                ], {'display': 'block', 'padding': '20px 60px 60px 60px', 'margin-top': '40px', 'scroll-margin-top': '75px'}, {}, True
            
            # Criar cabeçalho da seção
            header = [
                html.H3(f"{municipality}", 
                    style={
                        'color': 'white',
                        'margin-bottom': '20px',
                        'border-bottom': '2px solid #ffba3c',
                        'padding-bottom': '10px',
                        'font-weight': '700',
                        'font-size': '19px',
                        'font-family': 'roboto'
                    }),
                html.P(f"Encontrados {len(client_analyses)} clientes com potencial de compra", 
                    style={'color': '#cccccc', 'margin-bottom': '30px', 'font-size': '14px'})
            ]
            
            # Criar grid de cards em 3 colunas
            cards = []
            row = []
            
            for i, client in enumerate(client_analyses):
                row.append(html.Div(create_client_card(client, idx=i), style={'width': '32%', 'margin-bottom': '20px'}))
                
                if (i + 1) % 3 == 0 or i == len(client_analyses) - 1:
                    cards.append(html.Div(row, style={'display': 'flex', 'justifyContent': 'space-between'}))
                    row = []
            
            # Adicionar JavaScript para rolar até a seção
            script = html.Script("""
                setTimeout(function() {
                    document.getElementById('clientes-municipio-container').scrollIntoView({behavior: 'smooth'});
                }, 100);
            """)
            
            # Dicionário para armazenar dados de clientes para uso posterior
            client_data_dict = {str(i): client for i, client in enumerate(client_analyses)}
            
            # Retorna todos os outputs necessários
            return (
                header + cards + [script],  # children
                {'display': 'block', 'padding': '20px 60px 60px 60px', 'margin-top': '40px', 'scroll-margin-top': '75px'},  # style
                client_data_dict,  # all-client-data
                True  # scroll-trigger
            )
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())  # Isso imprimirá o stacktrace completo no console
            return [
                html.H3(f"Erro ao processar dados: {str(e)}", 
                       style={'color': 'white', 'text-align': 'center', 'margin-top': '30px'})
            ], {'display': 'block', 'padding': '20px 60px 60px 60px', 'margin-top': '40px', 'scroll-margin-top': '75px'}, {}, True

    @app.callback(
        Output('client-card-output', 'data'),
        [Input({'type': 'client-card', 'index': ALL}, 'n_clicks')],
        [State({'type': 'client-data', 'index': ALL}, 'data-client')]
    )
    def handle_client_card_click(n_clicks_list, client_data_list):
        if not n_clicks_list or not any(n for n in n_clicks_list if n):
            return dash.no_update
        
        # Identificar qual card foi clicado
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Obter o ID do componente que foi clicado
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            clicked_dict = json.loads(triggered_id)
            card_index = clicked_dict['index']
            
            # Encontrar o índice correspondente na lista de data-client
            for i, id_dict in enumerate(ctx.inputs_list[0]):
                if json.loads(id_dict['id'])['index'] == card_index:
                    if i < len(client_data_list) and client_data_list[i]:
                        # Retornar os dados do cliente que foram armazenados no elemento oculto
                        return json.loads(client_data_list[i])
        except Exception as e:
            print(f"Erro ao processar clique do card: {str(e)}")
        
        return dash.no_update

    return app  # Adicione esta linha para retornar o objeto app

def open_browser():
    Timer(3, lambda: webbrowser.open('http://127.0.0.1:8050/')).start()

def main():
    root = Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx")]
    )
    
    if not file_path or not os.path.isfile(file_path):
        print("Invalid file selection")
        return
        
    try:
        data = pd.read_excel(file_path)
        app = create_sales_dashboard(data)
        open_browser()
        app.run_server(host='127.0.0.1', port=8050, debug=False)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
