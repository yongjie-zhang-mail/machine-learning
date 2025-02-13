
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

from fastapi import FastAPI
import gradio as gr

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#####################################################################
## Final App Deployment

from frontend_block import get_demo

## If you do not have access to :8090, feel free to use /8090

demo = get_demo()
demo.queue()

logger.warning("Starting FastAPI app")
app = FastAPI()

app = gr.mount_gradio_app(app, demo, '/')

@app.route("/health")
async def health():
    return {"success": True}, 200
