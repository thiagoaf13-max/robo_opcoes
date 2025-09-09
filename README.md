# Robô Híbrido de Previsão (Google Sheets + Random Forest)

Este projeto implementa um robô que lê dados de uma planilha Google Sheets, treina um modelo de Machine Learning (RandomForest) de forma contínua e registra previsões futuras na própria planilha, combinando a confiança do modelo com estatísticas históricas por horário.

## Visão Geral
- **Entrada**: aba `Dados` de uma planilha Google Sheets (datas e horários de Sucesso/Falha).
- **Saída**: aba `Previsões` da mesma planilha (timestamp, texto da previsão, confiança e horário-alvo).
- **Modelo**: `RandomForestClassifier` com re-treinamento a cada ciclo.
- **Janela temporal**: próximos slots de tempo no múltiplo de `FREQ_MIN` minutos (padrão 5).
- **Hibridização**: média da confiança do modelo (ajustada por lags disponíveis) com a taxa histórica de acerto por horário.

## Arquivo principal
- `Robo hibrido.py`

Principais funções:
- `conectar_google_sheets()`: autentica via Service Account e abre as abas `Dados` e `Previsões`.
- `carregar_dados_google(aba_dados)`: lê registros e monta `DataFrame` tolerante a variações de nomes.
- `preparar_dados(df)`: cria colunas derivadas (hora, minuto, dia_semana, lags e slot HH:MM).
- `calcular_probabilidade_horario(df)`: taxa média de acerto por `slot`.
- `proximo_slot_5min(base_dt)`: retorna o próximo múltiplo de `FREQ_MIN` minutos.
- `registrar_previsao_google(...)`: grava a previsão na aba `Previsões`.
- `executar_robo()`: laço principal de leitura → treino → previsão → gravação → espera próximo slot.

## Requisitos
- Python 3.9+
- Bibliotecas:
  - pandas
  - gspread
  - google-auth (via `google.oauth2`)
  - scikit-learn

Você pode instalar via pip:

```bash
pip install pandas gspread google-auth scikit-learn
```

Opcionalmente, crie um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas gspread google-auth scikit-learn
```

## Credenciais e Google Sheets
1. Crie uma Service Account no Google Cloud e baixe o JSON (por exemplo `credenciais.json`).
2. Compartilhe sua planilha com o e-mail da Service Account (permissão de editor).
3. Ajuste no código, se necessário:
   - `SHEET_ID`: ID da planilha (parte da URL entre `/d/` e `/edit`).
   - `SHEET_NAME_DADOS`: nome da aba de entrada (padrão `Dados`).
   - `SHEET_NAME_PREVISOES`: nome da aba de saída (padrão `Previsões`).
   - `CREDENCIAIS_JSON`: caminho do arquivo JSON de credenciais.

### Estrutura esperada da aba `Dados`
O robô é tolerante a variações de nomes, mas espera colunas equivalentes a:
- `data`: data do registro (formato como `dd/mm/yyyy` ou similar).
- `Horário Sucesso`: horário (string) indicando quando houve acerto.
- `Horário Falha`: horário (string) indicando quando houve erro.

A coluna `resultado` é inferida: 1 se houver `Horário Sucesso`, 0 se `Horário Falha`, senão `None`.

## Parâmetros principais
Ajustáveis no topo de `Robo hibrido.py`:
- `NUM_LAGS = 12`: quantidade de lags de `resultado` usados como features.
- `LIMITE_CONFIANCA = 0.70`: mínimo para registrar previsão na aba `Previsões`.
- `ALERTA_CONFIANCA = 0.90`: marca visual de alerta quando a confiança for alta.
- `FREQ_MIN = 5`: granularidade dos slots (minutos) e tempo de espera entre ciclos.
- `TAMANHO_TAXA_MOVEL = 24`: tamanho da janela de memória local para taxa móvel (apenas informativo no log).

## Execução
Na raiz do projeto, execute:

```bash
python3 "Robo hibrido.py"
```

O robô:
- Conecta na planilha.
- Lê e prepara os dados.
- Treina um RandomForest em cada ciclo.
- Estima a próxima janela (ex.: de 5 em 5 minutos), faz a previsão e calcula a confiança híbrida.
- Se a confiança for >= `LIMITE_CONFIANCA`, grava a linha na aba `Previsões`.
- Aguarda até o próximo slot para repetir.

## Servidor (VPS Oracle Always Free)

Para disponibilizar um painel com botões Iniciar/Pausar/Parar e expor o robô 24/7:

1) Instalação

```bash
sudo apt update && sudo apt install -y python3-venv
cd /caminho/para/RoboOpcoes
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Credenciais e planilha

- Coloque `credenciais.json` na raiz do projeto.
- Compartilhe a planilha Google com o e-mail da Service Account.
- Ajuste `SHEET_ID` em `robo_core.py` se necessário.

3) Subir o servidor

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Abra no navegador: `http://SEU_IP:8000/`

4) Executar em produção (opcional)

Crie um serviço systemd para iniciar junto com a máquina:

```ini
[Unit]
Description=Robo Hibrido API
After=network.target

[Service]
Type=simple
WorkingDirectory=/caminho/para/RoboOpcoes
Environment="PATH=/caminho/para/RoboOpcoes/.venv/bin"
ExecStart=/caminho/para/RoboOpcoes/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Depois:

```bash
sudo systemctl daemon-reload
sudo systemctl enable robohibrido
sudo systemctl start robohibrido
```

5) Segurança

- Restrinja a porta com firewall se necessário.
- Não exponha `credenciais.json` publicamente.

## Como funciona a confiança
- **Confiança do modelo**: `predict_proba` do RandomForest.
- **Ajuste por lags**: reduz a confiança quando há poucos lags disponíveis no último registro.
- **Histórico por horário**: média de `resultado` por `slot` (HH:MM) no histórico.
- **Combinação**: média simples entre a confiança ajustada do modelo e a confiança histórica.

## Logs
O console exibirá mensagens com timestamps, incluindo avisos e erros comuns, por exemplo:
- Planilha não encontrada ou sem permissão.
- Dados vazios ou inválidos.
- Previsões registradas/ignoradas.

## Solução de problemas
- "Planilha não encontrada":
  - Confira `SHEET_ID` e se a Service Account tem acesso (compartilhe o documento com o e-mail da conta de serviço).
- "Erro ao conectar Google Sheets":
  - Valide caminho e conteúdo de `CREDENCIAIS_JSON`.
- Dados não aparecem:
  - Verifique nomes/variações das colunas e formatos de data e horário.
- Dependências:
  - Reinstale com `pip install --upgrade --force-reinstall pandas gspread google-auth scikit-learn`.

## Segurança
- Não commit o arquivo de credenciais em repositórios públicos.
- Restrinja o acesso da Service Account apenas ao necessário.

## Licença
Defina a licença do seu projeto (ex.: MIT).
