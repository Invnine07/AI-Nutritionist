import base64
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from zhipuai import ZhipuAI

app = FastAPI()

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 这里填入你刚刚复制的 API Key
client = ZhipuAI(api_key="aed5260b98344161a1a1ca2f3cf8115b.JUvDKhvspIcAUCpg") 

class ChatRequest(BaseModel):
    message: str
    user_info: str
    image_base64: str = None  # 新增：用来接收图片

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    print(f"📥 收到消息: {request.message}")
    
    system_prompt = f"""
    你是一个专业的AI营养师。你正在服务一位患者。
    患者的健康档案如下：{request.user_info}。
    请根据这些情况（尤其是糖尿病和高血压的饮食禁忌），给出专业的营养评估和改善建议。
    回答要简洁、温暖、专业，绝对不能给出危险的医疗建议。
    """

    try:
        # 如果前端传了图片，就用视觉大模型（glm-4v-flash）
        if request.image_base64:
            print("📷 检测到图片，使用多模态模型分析...")
            # 去掉 base64 的前缀，只保留数据
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
            # 如果没有图片，就正常聊天（glm-4-flash）
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message}
                ],
            )
            
        ai_reply = response.choices[0].message.content
        print(f"🤖 AI回复: {ai_reply}")
        return {"reply": ai_reply}
        
    except Exception as e:
        print(f"❌ 报错了: {e}")
        return {"reply": "抱歉，我的大脑暂时开小差了，请稍后再试。"}