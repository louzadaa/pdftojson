import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
import requests
from datetime import datetime
import pdfplumber
import io
import re
from typing import Dict, List, Any
import time

st.set_page_config(
    page_title="PDF to JSON",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chave da API OpenRouter
OPENROUTER_API_KEY = ""

class PDFExtractorAgents:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1)  # melhora alinhamento
                    if page_text:
                        text += f"\n=== PÁGINA {page_num} ===\n{page_text}\n"
            return text
        except Exception as e:
            st.error(f"Erro ao extrair texto do PDF: {str(e)}")
            return ""
    
    def call_openrouter_api(self, prompt: str, model: str = "qwen/qwen3-30b-a3b-instruct-2507") -> str:
        """Faz chamada para a API da OpenRouter"""
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 32768,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            st.error(f"Erro na chamada da API: {str(e)}")
            return ""
    
    def agent_valores_completo(self, pdf_text: str) -> Dict:
        prompt = f"""
Você é um agente especializado em extrair TODOS os VALORES e PLANOS de documentos de planos de saúde.

INSTRUÇÕES CRÍTICAS:
- Extraia TODAS as tabelas de preços encontradas no PDF.
- Inclua TODOS os planos, mesmo que tenham nomes semelhantes.
- Extraia TODAS as segmentações (ex.: 01-29 vidas, 30-99 vidas).
- Inclua TODOS os produtos e suas faixas etárias (00-18, 19-23, 24-28, ..., 59+).
- Identifique TODAS as acomodações (Ambulatorial, Enfermaria, Apartamento).
- NÃO omita nenhuma tabela, linha ou valor, mesmo que pareçam redundantes.
- Cada linha de tabela deve ser um objeto independente no array `planos_precos`.
- Indique claramente em qual página ou seção a informação foi encontrada.
- Utilize exatamente os valores monetários ou percentuais como aparecem no PDF.
- Extraia também observações, taxas e notas associadas aos preços.

TEXTO COMPLETO DO PDF:
{pdf_text}

Retorne APENAS um JSON válido seguindo esta estrutura exata:
{{
    "empresa": "nome_detectado_ou_empresa_desconhecida",
    "tipo_documento": "detectado_automaticamente",
    "regional": "regiao_detectada_ou_nao_especificada",
    "vigencia": {{"inicio": "data_ou_null", "fim": "data_ou_null"}},
    "planos_precos": [
        {{
            "id": "uuid",
            "produto": "nome_exato_do_plano",
            "tabela_origem": "nome_da_tabela",
            "posicao_na_pagina": "pagina_onde_encontrou",
            "empresa": "detectada",
            "tipo": "Ambulatorial_ou_Hospitalar_ou_Amb_Hosp_Obst",
            "categoria": "PME_ou_MEI_ou_ADESAO_ou_PJ",
            "segmentacao": "01_29_vidas_ou_30_99_vidas",
            "registro_ans": "numero_registro_se_encontrado",
            "acomodacao": "Enfermaria_Apartamento_ou_Nenhum",
            "descricao": "descricao_completa_do_plano",
            "valores_faixas": {{
                "00-18": "R$_valor_exato",
                "19-23": "R$_valor_exato",
                "24-28": "R$_valor_exato",
                "29-33": "R$_valor_exato",
                "34-38": "R$_valor_exato",
                "39-43": "R$_valor_exato",
                "44-48": "R$_valor_exato",
                "49-53": "R$_valor_exato",
                "54-58": "R$_valor_exato",
                "59+": "R$_valor_exato"
            }},
            "detalhes_adicionais": "informacoes_relevantes",
            "observacoes": "notas_importantes"
        }}
    ]
}}
"""
        response = self.call_openrouter_api(prompt)
        try:
            return json.loads(response)
        except Exception as e:
            st.error(f"Erro ao processar resposta do agente valores: {e}")
            return {"empresa": "empresa_desconhecida", "planos_precos": []}
    
    def agent_coparticipacao_completo(self, pdf_text: str) -> Dict:
        """Agente especializado em extrair TODAS as coparticipações e taxas"""
        import uuid
        prompt = f"""
Você é um agente especializado em extrair TODAS as COPARTICIPAÇÕES e TAXAS de documentos de planos de saúde.

INSTRUÇÕES CRÍTICAS:
- Extraia TODAS as tabelas de coparticipação e taxas encontradas.
- Inclua valores para:
  - Consultas (eletiva, telemedicina, urgência/emergência)
  - Exames (simples, especiais, complexos)
  - Terapias (simples, especiais)
  - Procedimentos ambulatoriais
  - Internações (enfermaria, apartamento)
  - Taxas administrativas (taxa de adesão, cadastro, etc.)
- Capture tanto valores fixos quanto percentuais (%)
- Se houver limites por valor (ex.: “30% limitado a R$ 146,20”), capture percentual e limite
- Relacione cada tabela de coparticipação aos produtos correspondentes (extraídos pelo agente de valores)
- Não omita nenhuma linha, mesmo que repetida

TEXTO COMPLETO DO PDF:
{pdf_text}

Retorne APENAS um JSON válido seguindo esta estrutura:
{{
    "tabelas_valores": [
        {{
            "id": "{str(uuid.uuid4())}",
            "tabela_origem": "identificador_da_tabela",
            "tipo": "coparticipacao_ambulatorial_ou_hospitalar_ou_taxas",
            "descricao": "descricao_detalhada_da_tabela",
            "aplica_produtos": ["lista_de_produtos_relacionados"],
            "valores": {{
                "consulta_eletiva": "valor_exato",
                "consulta_telemedicina": "valor_exato",
                "consulta_urgencia": "valor_exato",
                "exames_simples": "valor_ou_percentual",
                "exames_especiais": "valor_ou_percentual",
                "terapia_simples": "valor_ou_percentual",
                "terapia_especial": "valor_ou_percentual",
                "procedimentos_ambulatoriais": "valor_ou_percentual",
                "internacao_enfermaria": "valor_exato_ou_percentual",
                "internacao_apartamento": "valor_exato_ou_percentual",
                "taxa_adesao": "valor_exato_ou_null"
            }},
            "observacoes": "detalhes_importantes_extras"
        }}
    ]
}}
"""
        response = self.call_openrouter_api(prompt)
        try:
            return json.loads(response)
        except Exception as e:
            st.error(f"Erro ao processar resposta do agente coparticipação: {e}")
            return {"tabelas_valores": []}

    
    def agent_rede_credenciada_completo(self, pdf_text: str) -> Dict:

        prompt = f"""
Você é um agente especializado em extrair TODAS as informações sobre REDE CREDENCIADA de documentos de planos de saúde.

INSTRUÇÕES CRÍTICAS:
- Extraia TODOS os hospitais, clínicas, laboratórios e outros estabelecimentos citados
- Inclua todos os endereços, contatos e especialidades sempre que disponíveis
- Relacione cada estabelecimento aos produtos/plans em que está disponível (extraídos pelo agente de valores)
- Classifique cada item por tipo: hospital, clínica, laboratório, pronto-atendimento, consultório, etc.
- Indique se é “rede credenciada” ou “rede própria”
- Inclua cidade e região mencionadas
- Se não houver rede listada, retorne um JSON válido vazio, mas mantenha a estrutura
- Verifique TODAS as páginas, mesmo em notas de rodapé ou anexos

TEXTO COMPLETO DO PDF:
{pdf_text}

Retorne APENAS um JSON válido seguindo esta estrutura:
{{
    "informacoes_gerais": [
        {{
            "tipo": "hospitais_ou_clinicas_ou_laboratorios_ou_outros",
            "categoria": "rede_credenciada_ou_rede_propria",
            "regiao": "regiao_detectada",
            "lista": [
                {{
                    "nome": "nome_exato_do_estabelecimento",
                    "cidade": "cidade_detectada",
                    "detalhes": "informacoes_adicionais_e_especialidades",
                    "endereco": "endereco_completo_se_disponivel",
                    "contato": "telefone_email_ou_null",
                    "produtos_disponiveis": ["lista_de_produtos"],
                    "disponibilidade_por_produto": {{
                        "produto_x": "disponivel_ou_nao_disponivel",
                        "produto_y": "disponivel_ou_nao_disponivel"
                    }}
                }}
            ]
        }}
    ]
}}
"""
        response = self.call_openrouter_api(prompt)
        try:
            return json.loads(response)
        except Exception as e:
            st.error(f"Erro ao processar resposta do agente rede: {e}")
            return {"informacoes_gerais": []}

    def process_pdf_completo(self, pdf_file, page_number: int = 1) -> Dict:
        """Processa o PDF com EXTRAÇÃO COMPLETA usando todos os agentes"""
        # Extrair texto do PDF
        pdf_text = self.extract_text_from_pdf(pdf_file)
        
        if not pdf_text:
            return {}
        
        # Estrutura base do JSON
        result = {
            "pagina": page_number,
            "empresa": "empresa_desconhecida",
            "tipo_documento": "detectado_automaticamente",
            "regional": "nao_especificada",
            "vigencia": {"inicio": None, "fim": None},
            "tipo_pagina": "precos_ou_informativa",
            "planos_precos": [],
            "tabelas_valores": [],
            "informacoes_gerais": []
        }
        
        # Progress bar para os agentes
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Contadores para mostrar o progresso
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔍 Agente Valores Completo")
            agent1_status = st.empty()
            agent1_count = st.empty()
            agent1_status.info("⏳ Aguardando...")
        
        with col2:
            st.markdown("#### 💰 Agente Coparticipação Completo")
            agent2_status = st.empty()
            agent2_count = st.empty()
            agent2_status.info("⏳ Aguardando...")
        
        with col3:
            st.markdown("#### 🏢 Agente Rede Completa")
            agent3_status = st.empty()
            agent3_count = st.empty()
            agent3_status.info("⏳ Aguardando...")
        
        # Agente 1: Valores COMPLETOS
        status_text.text("🔍 Agente 1: Extraindo TODOS os valores e planos...")
        agent1_status.warning("🔄 Processando todas as tabelas...")
        progress_bar.progress(25)
        agent1_result = self.agent_valores_completo(pdf_text)
        
        planos_count = len(agent1_result.get("planos_precos", []))
        agent1_count.metric("Planos Extraídos", planos_count)
        agent1_status.success("✅ Concluído")
        
        # Agente 2: Coparticipações COMPLETAS
        status_text.text("💰 Agente 2: Extraindo TODAS as coparticipações...")
        agent2_status.warning("🔄 Processando todas as taxas...")
        progress_bar.progress(50)
        agent2_result = self.agent_coparticipacao_completo(pdf_text)
        
        copart_count = len(agent2_result.get("tabelas_valores", []))
        agent2_count.metric("Tabelas de Valores", copart_count)
        agent2_status.success("✅ Concluído")
        
        # Agente 3: Rede Credenciada COMPLETA
        status_text.text("🏢 Agente 3: Mapeando TODA a rede credenciada...")
        agent3_status.warning("🔄 Processando toda a rede...")
        progress_bar.progress(75)
        agent3_result = self.agent_rede_credenciada_completo(pdf_text)
        
        rede_count = sum(len(info.get("lista", [])) for info in agent3_result.get("informacoes_gerais", []))
        agent3_count.metric("Estabelecimentos", rede_count)
        agent3_status.success("✅ Concluído")
        
        # Combinar resultados
        status_text.text("📊 Combinando TODOS os resultados dos agentes...")
        progress_bar.progress(100)
        
        # Atualizar estrutura base com resultados dos agentes
        if agent1_result:
            result.update({
                "empresa": agent1_result.get("empresa", "empresa_desconhecida"),
                "tipo_documento": agent1_result.get("tipo_documento", "detectado_automaticamente"),
                "regional": agent1_result.get("regional", "nao_especificada"),
                "vigencia": agent1_result.get("vigencia", {"inicio": None, "fim": None}),
                "planos_precos": agent1_result.get("planos_precos", [])
            })
        
        if agent2_result:
            result["tabelas_valores"] = agent2_result.get("tabelas_valores", [])
        
        if agent3_result:
            result["informacoes_gerais"] = agent3_result.get("informacoes_gerais", [])
        
        # Limpar progress bar
        progress_bar.empty()
        status_text.empty()
        
        return result

def main():
    st.title("🧠 PDF to JSON")
    with st.sidebar:
        st.header("⚙️ Configurações")
        page_number = st.number_input("Número da Página Inicial", min_value=1, value=1)
    extractor = PDFExtractorAgents(OPENROUTER_API_KEY)
    
    uploaded_file = st.file_uploader(
        "📄 Faça upload do PDF do plano de saúde",
        type=['pdf'],
        help="Sistema otimizado para extrair TODAS as informações de documentos de planos de saúde"
    )
    
    if uploaded_file is not None:
        file_details = {
            "Nome": uploaded_file.name,
            "Tamanho": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Tipo": uploaded_file.type
        }
        
        st.info("📋 **Informações do Arquivo:**")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Botão para processar
        if st.button("🚀 EXTRAIR TUDO COM AGENTES IA", type="primary"):
            with st.spinner("Processando PDF com extração COMPLETA..."):
                
                # Processar PDF
                result = extractor.process_pdf_completo(uploaded_file, page_number)
                
                if result:
                    st.success("🎉 Extração COMPLETA concluída com sucesso!")
                    
                    # Mostrar resumo detalhado
                    st.markdown("### 📊 Resumo da Extração COMPLETA")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Empresa", result.get("empresa", "N/A"))
                    
                    with col2:
                        planos_count = len(result.get("planos_precos", []))
                        st.metric("Total de Planos", planos_count, delta=f"+{planos_count} extraídos")
                    
                    with col3:
                        tabelas_count = len(result.get("tabelas_valores", []))
                        st.metric("Tabelas de Valores", tabelas_count, delta=f"+{tabelas_count} extraídas")
                    
                    with col4:
                        rede_count = sum(len(info.get("lista", [])) for info in result.get("informacoes_gerais", []))
                        st.metric("Estabelecimentos", rede_count, delta=f"+{rede_count} mapeados")
                    
                    # Detalhamento por categoria
                    st.markdown("### 📈 Detalhamento da Extração")
                    
                    # Análise dos planos extraídos
                    if result.get("planos_precos"):
                        st.markdown("#### 🔍 Planos Extraídos por Categoria")
                        df_planos = pd.DataFrame(result["planos_precos"])
                        
                        if not df_planos.empty:
                            # Contador por produto
                            if 'produto' in df_planos.columns:
                                produtos_count = df_planos['produto'].value_counts()
                                st.write("**Distribuição por Produto:**")
                                for produto, count in produtos_count.items():
                                    st.write(f"- **{produto}**: {count} tabelas")
                            
                            # Contador por segmentação se disponível
                            if 'segmentacao' in df_planos.columns:
                                seg_count = df_planos['segmentacao'].value_counts()
                                st.write("**Distribuição por Segmentação:**")
                                for seg, count in seg_count.items():
                                    st.write(f"- **{seg}**: {count} planos")
                    
                    # Exibir JSON COMPLETO
                    st.markdown("### 📄 JSON COMPLETO Resultante")
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)
                    st.code(json_str, language='json')
                    
                    # Botões de download
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="⬇️ Baixar JSON COMPLETO",
                            data=json_str,
                            file_name=f"plano_saude_COMPLETO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Converter para DataFrame para análise
                        if result.get("planos_precos"):
                            df_planos = pd.DataFrame(result["planos_precos"])
                            csv = df_planos.to_csv(index=False)
                            st.download_button(
                                label="⬇️ CSV Planos COMPLETO",
                                data=csv,
                                file_name=f"planos_COMPLETO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        # Análise da rede credenciada
                        if result.get("informacoes_gerais"):
                            rede_data = []
                            for info in result["informacoes_gerais"]:
                                for item in info.get("lista", []):
                                    item_copy = item.copy()
                                    item_copy["tipo"] = info.get("tipo", "")
                                    item_copy["categoria"] = info.get("categoria", "")
                                    rede_data.append(item_copy)
                            
                            if rede_data:
                                df_rede = pd.DataFrame(rede_data)
                                csv_rede = df_rede.to_csv(index=False)
                                st.download_button(
                                    label="⬇️ CSV Rede COMPLETA",
                                    data=csv_rede,
                                    file_name=f"rede_credenciada_COMPLETA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    
                    # Análise estatística completa com Pandas e NumPy
                    if result.get("planos_precos"):
                        st.markdown("### 📈 Análise Estatística COMPLETA (Pandas & NumPy)")
                        
                        df_planos = pd.DataFrame(result["planos_precos"])
                        
                        if not df_planos.empty:
                            # Mostrar tabela completa
                            st.markdown("#### 📊 Tabela Completa de Planos")
                            st.dataframe(df_planos, use_container_width=True)
                            
                            # Análise de valores se disponível
                            if 'valores_faixas' in df_planos.columns:
                                st.markdown("#### 💰 Análise de Valores por Faixa Etária")
                                
                                # Extrair todos os valores numéricos
                                valores_numericos = []
                                for idx, row in df_planos.iterrows():
                                    if isinstance(row['valores_faixas'], dict):
                                        for faixa, valor in row['valores_faixas'].items():
                                            # Extrair números do valor
                                            numeros = re.findall(r'[\d,]+\.?\d*', str(valor))
                                            if numeros:
                                                try:
                                                    valor_num = float(numeros[0].replace(',', ''))
                                                    valores_numericos.append({
                                                        'produto': row.get('produto', 'N/A'),
                                                        'tipo': row.get('tipo', 'N/A'),
                                                        'segmentacao': row.get('segmentacao', 'N/A'),
                                                        'acomodacao': row.get('acomodacao', 'N/A'),
                                                        'tabela_origem': row.get('tabela_origem', 'N/A'),
                                                        'faixa_etaria': faixa,
                                                        'valor': valor_num
                                                    })
                                                except:
                                                    pass
                                
                                if valores_numericos:
                                    df_valores = pd.DataFrame(valores_numericos)
                                    st.dataframe(df_valores, use_container_width=True)
                                    
                                    # Estatísticas com NumPy
                                    valores_array = np.array(df_valores['valor'])
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Média", f"R$ {np.mean(valores_array):.2f}")
                                    with col2:
                                        st.metric("Mediana", f"R$ {np.median(valores_array):.2f}")
                                    with col3:
                                        st.metric("Menor Valor", f"R$ {np.min(valores_array):.2f}")
                                    with col4:
                                        st.metric("Maior Valor", f"R$ {np.max(valores_array):.2f}")
                                    
                                    st.write(f"**Desvio Padrão:** R$ {np.std(valores_array):.2f}")
                                    st.write(f"**Total de Valores Extraídos:** {len(valores_array)}")
                
                else:
                    st.error("❌ Erro ao processar o PDF. Tente novamente.")

if __name__ == "__main__":
    main()