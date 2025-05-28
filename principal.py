'''
ANS: https://dadosabertos.ans.gov.br/FTP/
IGR: https://dadosabertos.ans.gov.br/FTP/PDA/IGR/IGR_versao_2023/IGR.csv
OPERADORAS ATIVAS: https://dadosabertos.ans.gov.br/FTP/PDA/operadoras_de_plano_de_saude_ativas/Relatorio_cadop.csv
'''

from pyspark.sql import SparkSession, functions
from pyspark.sql.functions import col, count, avg, sum

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



spark = SparkSession.builder.master('local[*]').getOrCreate()

df_ativos = spark.read.csv("Relatorio_cadop.csv", inferSchema=True, header=True, sep=";", encoding="UTF-8")

df_reclamacoes = spark.read.csv("IGR.csv", inferSchema=True, header=True, sep=";", encoding="UTF-8")

df_model = df_ativos.filter(
        (col("Modalidade") == "Cooperativa Médica") | 
        (col("Modalidade") == "Medicina de Grupo") | 
        (col("Modalidade") == "Seguradora Especializada em Saúde")
)


def grafico(tabela_pd, anuncio, tema, total, name):

    plt.figure(figsize=(8,5))
    barras = plt.bar(tabela_pd[tema], tabela_pd[name], color="#98FB98")

    tema = tema.replace("_", " ").title()
    name = name.replace("_", " ").title()
    
    plt.title(f"{anuncio} {tema} - Total {total}")
    plt.xlabel(tema)
    plt.ylabel(name.title())
    plt.xticks(rotation=10)
    plt.grid(axis='y')

    for barra in barras: 
        valor = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, valor, int(valor), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def graficoCronologico(df_reclamacoes, registro_ans):
    
    df_reclamacoes = df_reclamacoes.withColumn(
        "ANO", functions.col("COMPETENCIA").substr(1, 4).cast("int")
    ).withColumn(
        "MES", functions.col("COMPETENCIA").substr(5, 2).cast("int")
    )

    df_reclamacoes = df_reclamacoes.filter(functions.col("REGISTRO_ANS") == registro_ans)
    
    df_reclamacoes_agrupado = df_reclamacoes.groupBy("ANO", "MES").agg(
        functions.sum("QTD_RECLAMACOES").alias("Total_Reclamacoes")
    ).orderBy("ANO", "MES")

    razao_social = df_reclamacoes.select("RAZAO_SOCIAL").distinct().collect()
    razao_social = razao_social[0]['RAZAO_SOCIAL']

    df_reclamacoes_agrupado_pd = df_reclamacoes_agrupado.toPandas()

    df_reclamacoes_agrupado_pd['Ano-Mês'] = df_reclamacoes_agrupado_pd['ANO'].astype(str) + '-' + df_reclamacoes_agrupado_pd['MES'].astype(str).str.zfill(2)

    plt.figure(figsize=(12, 6))
    plt.plot(df_reclamacoes_agrupado_pd['Ano-Mês'], df_reclamacoes_agrupado_pd['Total_Reclamacoes'], marker='o', color='g', label="Total de Reclamações")
    
    y_ticks = np.arange(0, 25, 4)
    plt.yticks(y_ticks)

    plt.title(f'Reclamações ao Longo do Tempo - {razao_social.title()}')
    plt.xlabel('Ano-Mês')
    plt.ylabel('Total de Reclamações')
    plt.xticks(rotation=45)

    ticks = df_reclamacoes_agrupado_pd['Ano-Mês'][::4]
    plt.xticks(ticks, rotation=45)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def ver_modalidade(df, df_tipos):
    
    name = "CONTAGEM"

    df_modalidade = df.groupBy("Modalidade").agg(count("*").alias(name))

    df_modalidade = df_modalidade.toPandas()

    total = df_modalidade["CONTAGEM"].sum()

    grafico(df_modalidade, anuncio="Divisão por ", tema="Modalidade", total=total, name=name)

    df_ativos_modalidade = df_tipos.groupBy("Modalidade").agg(functions.count("*").alias(name))

    df_ativos_modalidade = df_ativos_modalidade.toPandas()

    total_ativos = df_ativos_modalidade[name].sum()

    grafico(df_ativos_modalidade, anuncio="Quantidade Por ", tema="Modalidade", total=total_ativos, name=name)



def ativoXreclamacao(df_reclamacoes, df_tipos):


    df_tipos = df_tipos.withColumnRenamed("Razao_Social", "Razao_Social_1")
    df_reclamacoes = df_reclamacoes.withColumnRenamed("RAZAO_SOCIAL", "Razao_Social_2")

    df_f_grouped = df_tipos.groupBy("REGISTRO_ANS", "Razao_Social_1", "Modalidade").agg(
        functions.count("REGISTRO_ANS").alias("total_registros")
    ).orderBy("Razao_Social_1")

    juncao = df_f_grouped.join(df_reclamacoes, on="REGISTRO_ANS", how="inner").distinct()

    juncao = juncao.select(
        "REGISTRO_ANS", "Razao_Social_1", "Razao_Social_2", "COBERTURA", 
        "QTD_RECLAMACOES", "QTD_BENEFICIARIOS", "IGR", "Modalidade"
    )

    return juncao

def porBeneficiario():

    juncao = ativoXreclamacao(df_reclamacoes, df_model)
    
    #juncao.show(10,False)

    juncao = juncao.withColumn("IGR",functions.regexp_replace(functions.col("IGR"), "[^0-9.]", ""))  # REMOVE OS CARACTERES NÃO NUMERICOS, ISSO PQ O IGR ERA STRING DECIMAL, (STR 5,4) POR EX E RETORNAVA NULL SE CONVERTIDO PRA FLOAT OU INT DIRETAMENTE 

    juncao = juncao.withColumn("IGR", functions.col("IGR").cast("int"))

    #O CSV ANALISADO NESTA FUNÇÃO TEM VARIAS LINHAS PARA UM MESMO REGISTRO ANS, ENTÃO ESTOU RETIRANDO A MÉDIA PARA ANALISAR O DADO POR MÉDIA

    df_junt = juncao.groupBy("REGISTRO_ANS", "Razao_Social_1", "COBERTURA").agg(
        functions.avg("IGR").cast("int").alias("MEDIA_RECLAMACAO"),
        functions.avg("QTD_BENEFICIARIOS").cast("int").alias("MEDIA_BENEFICIARIOS")
    )

    df_junt = df_junt.filter((functions.col("COBERTURA") != "Exclusivamente odontológica"))

    #df_junt.printSchema()

    df_juntado = df_junt.withColumn(
        "Faixa_Beneficiarios",
        functions.when(functions.col("MEDIA_BENEFICIARIOS") <= 150000, "0 a 150k")
        .when((functions.col("MEDIA_BENEFICIARIOS") > 150000) & (functions.col("MEDIA_BENEFICIARIOS") <= 300000), "200k a 300k")
        .when((functions.col("MEDIA_BENEFICIARIOS") > 300000) & (functions.col("MEDIA_BENEFICIARIOS") <= 450000), "300k a 450k")
        .when((functions.col("MEDIA_BENEFICIARIOS") > 450000) & (functions.col("MEDIA_BENEFICIARIOS") <= 600000), "450k a 600k")
        .when((functions.col("MEDIA_BENEFICIARIOS") > 600000) & (functions.col("MEDIA_BENEFICIARIOS") <= 750000), "600k a 750k")
        .otherwise("750k+")
    )

    name = "TOTAL_REGISTROS"
    tota = df_juntado.count()

    df_segmentado = df_juntado.groupBy("Faixa_Beneficiarios").agg(
        functions.count("REGISTRO_ANS").alias(name)
    ).orderBy("Faixa_Beneficiarios")

    df_segmentado_pd = df_segmentado.toPandas()

    grafico(df_segmentado_pd, "Distribuição por ", "Faixa_Beneficiarios", total=tota, name=name)

    return df_juntado



def porReclamacao():

    #ESTA FUNÇÃO FAZ A MESMA QUE O CODIGO ACIMA, SÓ SEGMENTA PELA QUANTIDADE DE BENEFICIARIOS, MOSTRANDO GRAFICO DA MÉDIA POR RECLAMAÇÕES.

    df_juntado = porBeneficiario()

    df_juntado = df_juntado.filter(functions.col("Faixa_Beneficiarios") != "0 a 150k")

    df_juntado = df_juntado.withColumn(
        "MEDIA_RECLAMACAO_PERCENTUAL", (functions.col("MEDIA_RECLAMACAO") / 100).cast("int")
    )

    df_juntado = df_juntado.withColumn(
        "Faixa_Reclamacao",
        functions.when(functions.col("MEDIA_RECLAMACAO_PERCENTUAL") <= 20, "0 a 20%")
        .when((functions.col("MEDIA_RECLAMACAO_PERCENTUAL") > 20) & (functions.col("MEDIA_RECLAMACAO_PERCENTUAL") <= 40), "20 a 40%")
        .when((functions.col("MEDIA_RECLAMACAO_PERCENTUAL") > 40) & (functions.col("MEDIA_RECLAMACAO_PERCENTUAL") <= 60), "40 a 60%")
        .when((functions.col("MEDIA_RECLAMACAO_PERCENTUAL") > 60) & (functions.col("MEDIA_RECLAMACAO_PERCENTUAL") <= 80), "60 a 80%")
        .otherwise("80%+")
    )

    name = "Quantitativo de Empresas"
    tota_reclamacao = df_juntado.count()

    #PARA QUALQUER COLUNA QUE QUEIRA MOSTRAR AQUI, MUDE O GROUPY BY LA NA LINHA 76, O DF É RETURN DE OUTRA FUNÇÃO, INT TA CONDICIONADO

    #df_juntado.show(10)

    df_segmentado_reclamacao = df_juntado.groupBy("Faixa_Reclamacao").agg(
        functions.count("*").alias(name)
    ).orderBy("Faixa_Reclamacao")

    df_segmentado_reclamacao_pd = df_segmentado_reclamacao.toPandas()

    grafico(df_segmentado_reclamacao_pd, "Distribuição por ", "Faixa_Reclamacao", total=tota_reclamacao, name=name)

    #df_segmentado_reclamacao.show(5, False)

    df_baixa_reclamacao = df_juntado.filter((functions.col("Faixa_Reclamacao") == "0 a 20%") & (functions.col("COBERTURA") != "Exclusivamente odontológica")).orderBy("MEDIA_RECLAMACAO_PERCENTUAL")

    df_baixa_reclamacao.show(df_baixa_reclamacao.count())


def baixarCSV():
    print("Baixando todos os CSV's dependentes.")
    os.system("wget https://dadosabertos.ans.gov.br/FTP/PDA/operadoras_de_plano_de_saude_ativas/Relatorio_cadop.csv")
    time.sleep(2)
    os.system("wget https://dadosabertos.ans.gov.br/FTP/PDA/IGR/IGR_versao_2023/IGR.csv")
    local = os.listdir(os.getcwd())
    for doc in local: 
        if "IGR.csv" == doc:
            print(f"Indice de Reclamações baixado como {doc}")
        if "Relatorio_cadop.csv" == doc:
            print(f"Operadoras Ativas baixado como {doc}") 

def main():
    os.system("clear || cls")

    opcao = True
    while opcao:
        print("Painel Interativo para Análise, escolha:\n"
        "1) Ver modalidades.\n"
        "2) Gráfico de Beneficiarios e Reclamações.\n"
        "3) Gráfico Cronológico.\n"
        "4) Baixar CSV's.\n"
        "5) Sair."
        )

        try:
            opcao = str(input("\nEscolha: "))
            
            if opcao == "1":
                ver_modalidade(df_ativos, df_model)
            
            elif opcao == "2":
                porReclamacao()
            
            elif opcao == "3":
                registro_ans = input("Informe o registro ANS: ")
                graficoCronologico(df_reclamacoes, registro_ans)
                
            elif opcao == "4":
                baixarCSV()

            elif opcao == "5":    
                print("Saindo.")
                break
            
            else:
                print("\nOpção inválida. Tente novamente.")
        
        except Exception as e:
            print(f"Erro: {e}.")

if __name__ == "__main__":
    main()
