import os
import uuid
import asyncio
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import from deeplense_agent. If not available, we need to handle it or ensure we run in the right env.
from deeplense_agent.agent import create_agent

app = FastAPI(title="DeepLense AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    images: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None

# In-memory storage for sessions (for demo purposes)
# In production, use Redis or a database to serialize Pydantic-AI messages.
sessions = {}
agents = {}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = []
        # mock_mode=True means it doesn't need DeepLenseSim or colossus locally installed
        # but will still function and simulate data handling.
        # Assuming developers have the DEEPLENSE_PROVIDER and API KEY set in environment.
        agents[session_id] = create_agent(mock_mode=True)
        
    messages = sessions[session_id]
    agent = agents[session_id]
    
    num_sims_before = len(agent.deps.state.completed_simulations)
    
    try:
        result = await agent._agent.run(
            req.message,
            deps=agent.deps,
            message_history=messages
        )
        
        sessions[session_id] = result.all_messages()
        
        # Check if new simulations were run during this chat turn
        new_images = []
        num_sims_after = len(agent.deps.state.completed_simulations)
        if num_sims_after > num_sims_before:
            for sim in agent.deps.state.completed_simulations[num_sims_before:num_sims_after]:
                if sim.success:
                    for i, img in enumerate(sim.images):
                        new_images.append({
                            "index": i,
                            "width": img.width,
                            "height": img.height,
                            "base64_png": img.base64_png,
                            "simulation_id": sim.metadata.simulation_id if sim.metadata else "Unknown"
                        })
        
        # Get the current partial config
        current_config = None
        if agent.deps.state.current_request and agent.deps.state.current_request.config:
            current_config = agent.deps.state.current_request.config.model_dump()

        return ChatResponse(
            session_id=session_id,
            reply=result.output,
            images=new_images,
            config=current_config
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
