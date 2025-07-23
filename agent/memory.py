import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from models.models import Thread, Message, Event, Result
from agent.logger import get_logger, log_operation

# ============================================================================
# SISTEMA DE MEMÓRIA PERSISTENTE COM SQLITE
# ============================================================================

class AgentMemory:
    """
    Sistema de memória persistente thread-safe usando SQLite.
    Mantém contexto de conversas entre execuções do agente.
    Segue princípios funcionais com efeitos colaterais controlados.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self.logger = get_logger()
        self._lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """
        Inicializa o banco de dados e cria tabelas necessárias.
        Efeito colateral controlado na inicialização.
        """
        # Criar diretório se necessário
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            # Tabela principal de threads/conversas
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            # Tabela de mensagens individuais (para queries otimizadas)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    function_call TEXT,
                    name TEXT,
                    FOREIGN KEY (session_id) REFERENCES threads (session_id)
                )
            """)
            
            # Tabela de eventos para auditoria
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Índices para performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            
            conn.commit()
            
        self.logger.log_info("Banco de dados inicializado", db_path=str(self.db_path))
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager para conexões SQLite thread-safe.
        Garante fechamento adequado da conexão.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Acesso por nome de coluna
        try:
            yield conn
        finally:
            conn.close()
    
    def save_thread(self, thread: Thread) -> Result:
        """
        Salva uma thread completa no banco.
        Operação atômica com rollback em caso de erro.
        """
        try:
            with self._lock:
                with log_operation(self.logger, "save_thread", thread.session_id):
                    with self._get_connection() as conn:
                        # Serializar thread
                        thread_data = json.dumps(thread.to_dict(), ensure_ascii=False, default=str)
                        
                        # Upsert da thread principal
                        conn.execute("""
                            INSERT OR REPLACE INTO threads 
                            (session_id, data, created_at, updated_at, message_count)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            thread.session_id,
                            thread_data,
                            thread.created_at.isoformat(),
                            thread.updated_at.isoformat(),
                            len(thread.messages)
                        ))
                        
                        # Limpar mensagens antigas desta sessão
                        conn.execute("DELETE FROM messages WHERE session_id = ?", (thread.session_id,))
                        
                        # Inserir mensagens individuais
                        for message in thread.messages:
                            conn.execute("""
                                INSERT INTO messages 
                                (session_id, role, content, timestamp, function_call, name)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                thread.session_id,
                                message.role,
                                message.content,
                                message.timestamp.isoformat(),
                                json.dumps(message.function_call) if message.function_call else None,
                                message.name
                            ))
                        
                        conn.commit()
                        
                        # Log do evento de salvamento
                        save_event = Event(
                            type="system",
                            data={
                                "action": "thread_saved",
                                "session_id": thread.session_id,
                                "message_count": len(thread.messages)
                            },
                            session_id=thread.session_id
                        )
                        self._save_event(conn, save_event)
                        
                        return Result.ok({
                            "session_id": thread.session_id,
                            "message_count": len(thread.messages)
                        })
                        
        except Exception as e:
            error_msg = f"Erro ao salvar thread: {str(e)}"
            self.logger.log_error(error_msg, session_id=thread.session_id)
            return Result.error(error_msg)
    
    def load_thread(self, session_id: str) -> Result:
        """
        Carrega uma thread específica do banco.
        Retorna Result com Thread ou erro.
        """
        try:
            with self._lock:
                with log_operation(self.logger, "load_thread", session_id):
                    with self._get_connection() as conn:
                        cursor = conn.execute(
                            "SELECT data FROM threads WHERE session_id = ?",
                            (session_id,)
                        )
                        row = cursor.fetchone()
                        
                        if not row:
                            return Result.error(f"Thread não encontrada: {session_id}")
                        
                        # Deserializar thread
                        thread_data = json.loads(row['data'])
                        thread = Thread.from_dict(thread_data)
                        
                        return Result.ok(thread)
                        
        except Exception as e:
            error_msg = f"Erro ao carregar thread: {str(e)}"
            self.logger.log_error(error_msg, session_id=session_id)
            return Result.error(error_msg)
    
    def load_latest_thread(self) -> Result:
        """
        Carrega a thread mais recente (por updated_at).
        Útil para continuar conversas.
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT session_id FROM threads 
                        ORDER BY updated_at DESC 
                        LIMIT 1
                    """)
                    row = cursor.fetchone()
                    
                    if not row:
                        # Criar nova thread vazia
                        new_thread = Thread()
                        return Result.ok(new_thread)
                    
                    # Carregar thread encontrada
                    return self.load_thread(row['session_id'])
                    
        except Exception as e:
            error_msg = f"Erro ao carregar thread mais recente: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def list_sessions(self, limit: int = 50) -> Result:
        """
        Lista sessões existentes ordenadas por data de atualização.
        Função read-only para interface.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        session_id,
                        created_at,
                        updated_at,
                        message_count
                    FROM threads 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row['session_id'],
                        "created_at": row['created_at'],
                        "updated_at": row['updated_at'],
                        "message_count": row['message_count']
                    })
                
                return Result.ok(sessions)
                
        except Exception as e:
            error_msg = f"Erro ao listar sessões: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def delete_thread(self, session_id: str) -> Result:
        """
        Remove uma thread e todas suas mensagens.
        Operação destrutiva com confirmação via logs.
        """
        try:
            with self._lock:
                with log_operation(self.logger, "delete_thread", session_id):
                    with self._get_connection() as conn:
                        # Verificar se thread existe
                        cursor = conn.execute(
                            "SELECT message_count FROM threads WHERE session_id = ?",
                            (session_id,)
                        )
                        row = cursor.fetchone()
                        
                        if not row:
                            return Result.error(f"Thread não encontrada: {session_id}")
                        
                        message_count = row['message_count']
                        
                        # Deletar em cascata
                        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                        conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
                        conn.execute("DELETE FROM threads WHERE session_id = ?", (session_id,))
                        
                        conn.commit()
                        
                        # Log da deleção
                        delete_event = Event(
                            type="system",
                            data={
                                "action": "thread_deleted",
                                "session_id": session_id,
                                "deleted_messages": message_count
                            }
                        )
                        self._save_event(conn, delete_event)
                        
                        return Result.ok({
                            "session_id": session_id,
                            "deleted_messages": message_count
                        })
                        
        except Exception as e:
            error_msg = f"Erro ao deletar thread: {str(e)}"
            self.logger.log_error(error_msg, session_id=session_id)
            return Result.error(error_msg)
    
    def search_messages(self, query: str, session_id: Optional[str] = None, 
                       limit: int = 20) -> Result:
        """
        Busca mensagens por conteúdo.
        Função de pesquisa read-only.
        """
        try:
            with self._get_connection() as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT session_id, role, content, timestamp, name
                        FROM messages 
                        WHERE session_id = ? AND content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (session_id, f"%{query}%", limit))
                else:
                    cursor = conn.execute("""
                        SELECT session_id, role, content, timestamp, name
                        FROM messages 
                        WHERE content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (f"%{query}%", limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "session_id": row['session_id'],
                        "role": row['role'],
                        "content": row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                        "timestamp": row['timestamp'],
                        "name": row['name']
                    })
                
                return Result.ok(results)
                
        except Exception as e:
            error_msg = f"Erro na busca: {str(e)}"
            self.logger.log_error(error_msg, query=query)
            return Result.error(error_msg)
    
    def _save_event(self, conn: sqlite3.Connection, event: Event) -> None:
        """
        Salva evento no banco (método interno).
        Usado para auditoria e debugging.
        """
        conn.execute("""
            INSERT INTO events (session_id, event_type, data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            event.session_id,
            event.type,
            json.dumps(event.data, ensure_ascii=False, default=str),
            event.timestamp.isoformat()
        ))
    
    def get_events(self, session_id: Optional[str] = None, 
                   event_type: Optional[str] = None, limit: int = 100) -> Result:
        """
        Recupera eventos para análise.
        Função read-only para debugging e métricas.
        """
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                events = []
                for row in cursor.fetchall():
                    try:
                        event_data = json.loads(row['data'])
                        events.append({
                            "id": row['id'],
                            "session_id": row['session_id'],
                            "event_type": row['event_type'],
                            "data": event_data,
                            "timestamp": row['timestamp']
                        })
                    except json.JSONDecodeError:
                        # Skip malformed events
                        continue
                
                return Result.ok(events)
                
        except Exception as e:
            error_msg = f"Erro ao recuperar eventos: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Result:
        """
        Limpa dados antigos para manter o banco otimizado.
        Operação de manutenção com logging detalhado.
        """
        try:
            cutoff_date = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_to_keep)
            
            with self._lock:
                with log_operation(self.logger, "cleanup_old_data"):
                    with self._get_connection() as conn:
                        # Contar dados a serem removidos
                        cursor = conn.execute("""
                            SELECT COUNT(*) as count FROM threads 
                            WHERE updated_at < ?
                        """, (cutoff_date.isoformat(),))
                        old_threads = cursor.fetchone()['count']
                        
                        cursor = conn.execute("""
                            SELECT COUNT(*) as count FROM events 
                            WHERE timestamp < ?
                        """, (cutoff_date.isoformat(),))
                        old_events = cursor.fetchone()['count']
                        
                        if old_threads == 0 and old_events == 0:
                            return Result.ok({
                                "cleaned_threads": 0,
                                "cleaned_events": 0,
                                "message": "Nenhum dado antigo encontrado"
                            })
                        
                        # Remover dados antigos
                        conn.execute("""
                            DELETE FROM messages WHERE session_id IN (
                                SELECT session_id FROM threads WHERE updated_at < ?
                            )
                        """, (cutoff_date.isoformat(),))
                        
                        conn.execute("""
                            DELETE FROM threads WHERE updated_at < ?
                        """, (cutoff_date.isoformat(),))
                        
                        conn.execute("""
                            DELETE FROM events WHERE timestamp < ?
                        """, (cutoff_date.isoformat(),))
                        
                        # Vacuum para otimizar espaço
                        conn.execute("VACUUM")
                        conn.commit()
                        
                        return Result.ok({
                            "cleaned_threads": old_threads,
                            "cleaned_events": old_events,
                            "cutoff_date": cutoff_date.isoformat()
                        })
                        
        except Exception as e:
            error_msg = f"Erro na limpeza de dados: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def get_stats(self) -> Result:
        """
        Retorna estatísticas do banco de dados.
        Função read-only para métricas e monitoramento.
        """
        try:
            with self._get_connection() as conn:
                # Contar threads e mensagens
                cursor = conn.execute("SELECT COUNT(*) as count FROM threads")
                total_threads = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
                total_messages = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM events")
                total_events = cursor.fetchone()['count']
                
                # Thread mais recente
                cursor = conn.execute("""
                    SELECT updated_at FROM threads 
                    ORDER BY updated_at DESC LIMIT 1
                """)
                latest_row = cursor.fetchone()
                latest_activity = latest_row['updated_at'] if latest_row else None
                
                # Tamanho do arquivo de banco
                db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
                db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
                
                return Result.ok({
                    "total_threads": total_threads,
                    "total_messages": total_messages,
                    "total_events": total_events,
                    "latest_activity": latest_activity,
                    "db_size_mb": db_size_mb,
                    "db_path": str(self.db_path)
                })
                
        except Exception as e:
            error_msg = f"Erro ao obter estatísticas: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)

# ============================================================================
# MEMORY FACTORY E UTILITÁRIOS
# ============================================================================

from datetime import timedelta

_global_memory: Optional[AgentMemory] = None

def get_memory(db_path: str = "memory.db") -> AgentMemory:
    """
    Retorna instância global de memória (singleton pattern).
    Thread-safe lazy initialization.
    """
    global _global_memory
    
    if _global_memory is None or str(_global_memory.db_path) != db_path:
        _global_memory = AgentMemory(db_path)
    
    return _global_memory

def create_empty_thread(session_id: Optional[str] = None) -> Thread:
    """
    Factory function para criar thread vazia.
    Função pura de criação.
    """
    if session_id is None:
        session_id = f"session_{datetime.utcnow().timestamp()}"
    
    return Thread(session_id=session_id)

def backup_memory(source_db: str = "memory.db", backup_path: Optional[str] = None) -> Result:
    """
    Cria backup do banco de memória.
    Operação de segurança com verificação de integridade.
    """
    try:
        source_path = Path(source_db)
        if not source_path.exists():
            return Result.error(f"Banco fonte não encontrado: {source_db}")
        
        if backup_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"memory_backup_{timestamp}.db"
        
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copiar arquivo
        import shutil
        shutil.copy2(source_path, backup_path)
        
        # Verificar integridade do backup
        with sqlite3.connect(str(backup_path)) as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            
            if integrity != "ok":
                backup_path.unlink()  # Remover backup corrompido
                return Result.error(f"Backup corrompido: {integrity}")
        
        return Result.ok({
            "source": str(source_path),
            "backup": str(backup_path),
            "size_mb": round(backup_path.stat().st_size / (1024 * 1024), 2)
        })
        
    except Exception as e:
        return Result.error(f"Erro no backup: {str(e)}")
