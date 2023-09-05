## Multi-docs-query

### 执行步骤
1. 在 `chat_uber.py` 中配置你的OPENAI_API_KEY
2.  确保你的机器可以连接到OPENAI的服务器
3. `python3 chat_uber.py`

### 过程解释
1.  程序读取了位于 `data/UBER` 文件夹下四年的UBER的财务表现综合性报告
2. 对报告中的字段分块（512个词为一个字段），同时调用`openai`接口将内容创建`embedding`（可替换为其他大模型），并将`embedding`序列化到`chromadb`中（可替换为`Milvus`）
3. 读取`embedding`，构建Composable graph query
4. Query document