import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from zhipuai import ZhipuAI
import base64

app = FastAPI()

# 从环境变量中读取 API Key。如果本地没有配置环境变量，默认给个空字符串
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ 注意：把下面引号里的内容换成你自己的 API Key！
client = ZhipuAI(api_key=ZHIPU_API_KEY)

class ChatRequest(BaseModel):
    message: str
    user_info: str
    image_base64: str = None

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    print(f"📥 收到消息: {request.message}")
    
    # 注意这里的缩进，必须在函数内部
    system_prompt = f"""
    你是一位拥有丰富临床经验的专业注册营养师。你正在服务一位患有慢病的患者。
    患者的健康档案如下：{request.user_info}。

    【核心指令】
    请根据患者的疾病史（如糖尿病、高血压），对用户提供的食物（文字或图片）进行深度分析，并**严格按照以下四个模块**使用 Markdown 格式输出：
    
    ### 1. 识别与热量估算
    - 识别出的食物名称。
    - **预估分量与热量**：估算该食物的重量（克）和总热量（大卡/kcal）。
    
    ### 2. 核心营养素分析
    - 重点指出该食物中与患者疾病高度相关的成分（例如：糖尿病患者需关注**碳水化合物/糖分/升糖指数(GI)**；高血压患者需关注**钠含量/脂肪**）。
    
    ### 3. 针对患病人群的建议
    - 结合其疾病史，明确指出“能不能吃”、“如果吃应该怎么控制量”、“搭配什么吃更好”。
    
    ### 4. 饮食补救与改善措施
    - 如果该食物不健康或热量超标，请给出**具体的补救措施**（例如：下一餐应该减少多少主食、建议进行何种强度的运动来消耗多余热量等）。

    【约束条件】
    - 语气必须专业、严谨、温暖。
    - 绝对不能给出任何具体的药物建议或危险的医疗方案，仅限饮食与生活方式干预。
    """

    try:
        if request.image_base64:
            print("📷 检测到图片，使用视觉模型 glm-4v-flash...")
            base64_data = request.image_base64.split(',')[1]
            
            response = client.chat.completions.create(
                model="glm-4v-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": request.message},
                        {"type": "image_url", "image_url": {"url": base64_data}}
                    ]}
                ]
            )
        else:
            print("💬 纯文本对话，使用模型 glm-4-flash...")
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message}
                ]
            )
            
        ai_reply = response.choices[0].message.content
        print(f"🤖 AI回复: {ai_reply}")
        return {"reply": ai_reply}
        
    except Exception as e:
        print(f"❌ 报错了: {e}")
        return {"reply": "抱歉，我的大脑暂时开小差了，请稍后再试。"}