# -*- coding: utf-8 -*-
# shape 0 linha
# shape 1 coluna
import cv2 as cv
import numpy as np


def binariza_assinatura(assinatura):
    for i in range(assinatura.shape[0]):
        for j in range(assinatura.shape[1]):
            if assinatura[i][j] < 10:
                assinatura[i][j] = 255
            else:
                assinatura[i][j] = 0
    return assinatura


def recorta_lateral_assinatura(assinatura):
    x1 = None
    for j in range(assinatura.shape[1]):
        for i in range(assinatura.shape[0]):
            if (assinatura[i][j] != 255):
                x1 = j
                break
        if x1 is not None:
            break
    x2 = None
    for j in range(assinatura.shape[1]-1, 0, -1):
        for i in range(assinatura.shape[0]-1, 0, -1):
            if (assinatura[i][j] != 255):
                x2 = j
                break
        if x2 is not None:
            break

    assinatura = assinatura[0:assinatura.shape[0], x1:x2]
    return assinatura


def extrai_caracteristicas(assinatura):
    caracteristicas = [0] * assinatura.shape[1]
    for j in range(assinatura.shape[1]):
        for i in range(assinatura.shape[0]):
            if assinatura[i][j] == 0:
                caracteristicas[j] += 1
    return caracteristicas


def padroniza_vetor(caracteristicas, tamanho):
    aux = []
    for i in range(tamanho):
        if i < len(caracteristicas):
            aux.append(caracteristicas[i])
        else:
            aux.append(0)
    return aux


def reduz_ruidos(assinatura):
    assinatura = cv.medianBlur(assinatura, 3)
    assinatura = cv.GaussianBlur(assinatura, (5, 5), 0)
    return assinatura


def salva_vetor_treino(caracteristicas):
    with open('../data/vetores/treino/vetores.txt', 'a') as arq_vetor:
        arq_vetor.write(str(caracteristicas)+',\n')


def salva_vetor_teste(caracteristicas):
    with open('../data/vetores/teste/vetores.txt', 'a') as arq_vetor:
        arq_vetor.write(str(caracteristicas)+',\n')


def lista_arquivos_treino(num_pastas, imagens_por_pasta):
    nomes = []
    for pasta in range(num_pastas):
        for imagem in range(imagens_por_pasta):
            nomes.append(
                '../data/imagens/treino/'
                f'{pasta+1}'
                '/0'
                f'{imagem+1}'
                '.PNG'
            )
    return nomes


def lista_arquivos_teste(num_imagens):
    nomes = []
    for imagem in range(num_imagens):
        nomes.append(
            '../data/imagens/teste/'
            f'{imagem+1}'
            '.PNG'
        )
    return nomes


if __name__ == "__main__":
    arquivos_treino = lista_arquivos_treino(28, 5)
    for nome_arquivo in arquivos_treino:
        print(nome_arquivo)
        assinatura = cv.imread(nome_arquivo, 0)
        assinatura = reduz_ruidos(assinatura)
        assinatura = binariza_assinatura(assinatura)
        assinatura = recorta_lateral_assinatura(assinatura)
        caracteristicas = extrai_caracteristicas(assinatura)
        caracteristicas = padroniza_vetor(caracteristicas, 300)
        salva_vetor_treino(caracteristicas)

    arquivos_teste = lista_arquivos_teste(32)
    for nome_arquivo in arquivos_teste:
        print(nome_arquivo)
        assinatura = cv.imread(nome_arquivo, 0)
        assinatura = reduz_ruidos(assinatura)
        assinatura = binariza_assinatura(assinatura)
        assinatura = recorta_lateral_assinatura(assinatura)
        caracteristicas = extrai_caracteristicas(assinatura)
        caracteristicas = padroniza_vetor(caracteristicas, 300)
        salva_vetor_teste(caracteristicas)
