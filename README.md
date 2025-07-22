# Agente funcional com contexto persistente

Este projeto demonstra um agente inteligente que mantém conversas com memória de longo prazo, executa funções estruturadas, registra logs confiáveis e utiliza boas práticas de programação funcional e tipagem forte.

---

## Principais Características

- **Memória persistente (SQLite)** para manter contexto entre execuções
- **Tool use via OpenAI Function Calls** para ações estruturadas e seguras
- **Logs estruturados** para monitoramento confiável
- **Engenharia de contexto com dataclasses, type hints e YAML**
- **Loop autônomo com retry/fallback** para resiliência
- **Tratamento dos resultados das funções como novos eventos**
- **Teoria de tipos e programação funcional** para segurança e previsibilidade

---

![Arquitetura](images/ChatGPT%20Image%2022%20de%20jul.%20de%202025%2C%2008_14_22.png)

...

## Estrutura de Pastas

```
context-agent-llm/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── tools.py
│   ├── memory.py
│   ├── context.py
│   └── logger.py
│
├── models/
│   ├── __init__.py
│   └── models.py
│
└── tests/
    ├── __init__.py
    └── test_agent.py
```

---

## Como rodar (mini tutorial)

1. **Clone este repositório:**
    ```bash
    git clone https://github.com/seu-usuario/context-agent-llm.git
    cd context-agent-llm
    ```

2. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Obtenha uma chave da [OpenRouter](https://openrouter.ai/) (grátis) e coloque no arquivo `.env`:**
    ```
    OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
    ```

4. **Execute o agente:**
    ```bash
    python main.py
    ```

> **Dica:** Altere o modelo LLM em `agent/agent.py` (ex: `"mistralai/mistral-7b-instruct"`, `"meta-llama/llama-3-8b-instruct"`, etc).

---

## Estrutura dos arquivos

- `models.py` — Modelos imutáveis e tipados (eventos, mensagens, resultados)
- `context.py` — Conversão imutável do contexto para prompt YAML
- `memory.py` — Camada de persistência SQLite
- `tools.py` — Funções/tool use simuladas e seguras
- `logger.py` — Logging estruturado para observabilidade
- `agent.py` — Loop autônomo, integração LLM, retries, evolução de contexto
- `main.py` — Script principal de execução

---

## Conceitos aplicados

| Conceito                                                 | Aplicação                                                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Teoria de tipos (type hints + Literal)**               | Segurança e validação dos dados                                                                  |
| **Imutabilidade (frozen dataclasses)**                   | Estado previsível, sem efeitos colaterais                                                        |
| **Funções puras**                                        | Manipulação de contexto sem alterar estado externo                                               |
| **Máquina de estados pura**                              | Evolução de `Thread` como sequência de eventos                                                   |
| **Efeitos colaterais controlados (logs/persistência)**   | Logging e salvamento explícitos                                                                  |
| **Retry, fallback e resiliência**                        | Robustez no ciclo autônomo do agente                                                             |

---

## Licença

MIT
