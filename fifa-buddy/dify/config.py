"""
Dify 客户端配置文件
"""

# Dify API 配置
DIFY_CONFIG = {
    # API 密钥（请替换为您的实际 API 密钥）
    # "api_key": "your-dify-api-key-here",
    "api_key": "app-aEyisVLt3LuejiV2lyE2XTu0",
    
    # API 基础 URL（如果使用自部署的 Dify，请修改此 URL）
    # "base_url": "https://api.dify.ai/v1",
    "base_url": "https://api.dify.ai/v1",
    
    # 默认用户标识
    "default_user": "default_user",
    
    # 请求超时时间（秒）
    "timeout": 30,
    
    # 重试次数
    "max_retries": 3
}

# 预设的输入变量模板
INPUT_TEMPLATES = {
    "simple_chat": {},
    
    "with_context": {
        "context": "这里可以提供上下文信息",
        "language": "zh-CN"
    },
    
    "fifa_analysis": {
        "match_type": "世界杯",
        "team1": "巴西",
        "team2": "阿根廷",
        "analysis_focus": "战术分析"
    },

    "fifa_buddy_event": {
        "message_type": "event",
        "message": "{\"match_id\": 146248, \"match_run_time_in_ms\": 1747295, \"match_run_time\": \"00:29:07\", \"match_time_in_ms\": 1747295, \"event_id\": 2221, \"relevant_event_id\": 2210, \"team_id\": 1935290, \"from_player_id\": 483484, \"player_seq_id\": 737, \"event_order\": 4, \"half_time\": 1, \"category\": \"in_possession\", \"event_type\": \"possession_outcome\", \"event\": \"goal\", \"side\": \"l\", \"x\": 1.000476241, \"x_mirrored\": 1.000476241, \"y\": 0.548382342, \"y_mirrored\": 0.548382342, \"action_type\": \"\", \"to_player_id\": \"\", \"sequence_type\": \"\", \"outcome\": \"possession_complete\", \"outcome_additional\": \"\", \"opposition_touch\": \"\", \"body_type\": \"\", \"direction\": \"\", \"pressure\": \"\", \"style\": \"\", \"style_additional\": \"\", \"frame_location\": \"\", \"game_state\": \"\", \"game_period\": \"\", \"game_period_additional\": \"\", \"game_involvement\": \"\", \"origin\": \"pass\", \"origin_additional\": \"assist\", \"save_type\": \"\", \"save_detail\": \"\", \"stance\": \"\", \"x_frame\": \"\", \"y_frame\": \"\", \"movement\": \"\", \"offering_to_receive_total_units\": \"\", \"line_break_direction\": \"\", \"line_break_outcome\": \"\", \"team_shape\": \"\", \"team_unit\": \"\", \"team_units_broken\": \"\", \"total_team_units\": \"\", \"event_end_time_in_ms\": \"\", \"x_location_start\": \"\", \"x_location_start_mirrored\": \"\", \"x_location_end\": \"\", \"x_location_end_mirrored\": \"\", \"y_location_start\": \"\", \"y_location_start_mirrored\": \"\", \"y_location_end\": \"\", \"y_location_end_mirrored\": \"\", \"version\": \"v1.2\", \"team_name\": \"CHELSEA FC\", \"from_player_name\": \"COLE PALMER\", \"from_player_shirt_number\": 10, \"to_player_name\": \"\", \"to_player_shirt_number\": \"\"}"
    },

    "fifa_buddy_other_test": {
        "message_type": "other",
        "message": "{\"match_id\": 146248, \"match_run_time_in_ms\": 1747295, \"match_run_time\": \"00:29:07\", \"match_time_in_ms\": 1747295, \"event_id\": 2221, \"relevant_event_id\": 2210, \"team_id\": 1935290, \"from_player_id\": 483484, \"player_seq_id\": 737, \"event_order\": 4, \"half_time\": 1, \"category\": \"in_possession\", \"event_type\": \"possession_outcome\", \"event\": \"goal\", \"side\": \"l\", \"x\": 1.000476241, \"x_mirrored\": 1.000476241, \"y\": 0.548382342, \"y_mirrored\": 0.548382342, \"action_type\": \"\", \"to_player_id\": \"\", \"sequence_type\": \"\", \"outcome\": \"possession_complete\", \"outcome_additional\": \"\", \"opposition_touch\": \"\", \"body_type\": \"\", \"direction\": \"\", \"pressure\": \"\", \"style\": \"\", \"style_additional\": \"\", \"frame_location\": \"\", \"game_state\": \"\", \"game_period\": \"\", \"game_period_additional\": \"\", \"game_involvement\": \"\", \"origin\": \"pass\", \"origin_additional\": \"assist\", \"save_type\": \"\", \"save_detail\": \"\", \"stance\": \"\", \"x_frame\": \"\", \"y_frame\": \"\", \"movement\": \"\", \"offering_to_receive_total_units\": \"\", \"line_break_direction\": \"\", \"line_break_outcome\": \"\", \"team_shape\": \"\", \"team_unit\": \"\", \"team_units_broken\": \"\", \"total_team_units\": \"\", \"event_end_time_in_ms\": \"\", \"x_location_start\": \"\", \"x_location_start_mirrored\": \"\", \"x_location_end\": \"\", \"x_location_end_mirrored\": \"\", \"y_location_start\": \"\", \"y_location_start_mirrored\": \"\", \"y_location_end\": \"\", \"y_location_end_mirrored\": \"\", \"version\": \"v1.2\", \"team_name\": \"CHELSEA FC\", \"from_player_name\": \"COLE PALMER\", \"from_player_shirt_number\": 10, \"to_player_name\": \"\", \"to_player_shirt_number\": \"\"}"
    }

    
}

# 文件上传配置（用于支持 Vision 功能的模型）
FILE_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".txt"],
    "image_extensions": [".jpg", ".jpeg", ".png", ".gif"]
}
