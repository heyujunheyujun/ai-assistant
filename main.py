from typing import List, Optional
from pathlib import Path
import os
from functools import lru_cache
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentRetrievalSystem:
    """文档检索系统，负责文档的分块、向量化、存储和检索"""
    
    def __init__(self, doc_path: str, model_name: str = "shibing624/text2vec-base-chinese"):
        self.doc_path = Path(doc_path)
        self.embedding_model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        self.chromadb_client = chromadb.EphemeralClient()
        self.collection = None
        self.chunks = []
        
    def split_into_chunks(self, chunk_separator: str = "\n\n") -> List[str]:
        """将文档分割成片段"""
        try:
            with open(self.doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = [chunk.strip() for chunk in content.split(chunk_separator) if chunk.strip()]
            logger.info(f"文档已分割为 {len(chunks)} 个片段")
            return chunks
        except FileNotFoundError:
            logger.error(f"文档文件未找到: {self.doc_path}")
            raise
        except Exception as e:
            logger.error(f"读取文档时出错: {e}")
            raise
    
    def embed_chunk(self, chunk: str) -> List[float]:
        """将文本片段转换为向量"""
        try:
            embedding = self.embedding_model.encode(chunk, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"向量化片段时出错: {e}")
            raise
    
    def save_embeddings(self, chunks: List[str], batch_size: int = 100) -> None:
        """批量保存片段和向量到数据库"""
        self.collection = self.chromadb_client.get_or_create_collection(
            name="documents",
            embedding_function=None
        )
        
        self.chunks = chunks
        
        # 批量处理以提高效率
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = [self.embed_chunk(chunk) for chunk in batch_chunks]
            batch_ids = [str(j) for j in range(i, i+len(batch_chunks))]
            
            self.collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                ids=batch_ids
            )
            logger.info(f"已保存批次 {i//batch_size + 1}，共 {len(batch_chunks)} 个片段")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """根据查询召回相关片段"""
        if not self.collection:
            logger.error("向量数据库未初始化")
            return []
        
        try:
            query_embedding = self.embed_chunk(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"检索时出错: {e}")
            return []
    
    def rerank(self, query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
        """对召回的片段进行重排序"""
        if not retrieved_chunks:
            return []
        
        try:
            pairs = [(query, chunk) for chunk in retrieved_chunks]
            scores = self.cross_encoder.predict(pairs)
            
            scored_chunks = list(zip(retrieved_chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            return [chunk for chunk, _ in scored_chunks][:top_k]
        except Exception as e:
            logger.error(f"重排序时出错: {e}")
            return retrieved_chunks[:top_k]
    
    @lru_cache(maxsize=128)
    def generate_answer(self, query: str, top_k_retrieve: int = 5, top_k_rerank: int = 3) -> str:
        """生成最终答案（带缓存）"""
        try:
            # 召回
            retrieved_chunks = self.retrieve(query, top_k_retrieve)
            logger.info(f"召回了 {len(retrieved_chunks)} 个片段")
            
            # 重排
            reranked_chunks = self.rerank(query, retrieved_chunks, top_k_rerank)
            logger.info(f"重排序后保留 {len(reranked_chunks)} 个片段")
            
            if not reranked_chunks:
                return "未找到相关内容。"
            
            # 构建提示词
            prompt = self._build_prompt(query, reranked_chunks)
            
            # 调用LLM
            return self._call_llm(query, prompt)
            
        except Exception as e:
            logger.error(f"生成答案时出错: {e}")
            return f"生成答案失败: {str(e)}"
    
    def _build_prompt(self, query: str, chunks: List[str]) -> str:
        """构建提示词"""
        context = "\n\n".join(chunks)
        return f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题：{query}

相关片段：
{context}

请基于上诉内容作答，不要编造信息。"""
    
    def _call_llm(self, query: str, prompt: str) -> str:
        """调用LLM API"""
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )
        
        try:
            response = client.chat.completions.create(
                model="kimi-k2.5",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=1,  # 添加温度参数控制创造性
                max_tokens=1000   # 限制最大输出长度
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise

def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("未设置OPENAI_API_KEY环境变量")
        return
    
    # 初始化检索系统
    try:
        retriever = DocumentRetrievalSystem("doc.md")
        
        # 分割文档并保存向量
        chunks = retriever.split_into_chunks()
        retriever.save_embeddings(chunks)
        
        # 生成答案
        answer = retriever.generate_answer('哆啦A梦使用的3个秘密道具分别是什么？')
        print(answer)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()