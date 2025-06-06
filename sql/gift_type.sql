CREATE TABLE `gift_type` (
  `code` int NOT NULL AUTO_INCREMENT COMMENT '奖品类型编码',
  `name` varchar(200) COLLATE utf8mb4_general_ci NOT NULL COMMENT '奖品名称',
  `sub_name` varchar(200) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '奖品副名称',
  `gift_desc` varchar(200) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '奖品描述',
  `gift_spec` varchar(200) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '奖品说明',
  `pic` varchar(500) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '奖品图片',
  `islimit` int DEFAULT NULL COMMENT '是否限制数量(0:否,1:是)',
  `total_limit` int DEFAULT NULL COMMENT '总量限制',
  `remaining_count` int DEFAULT NULL COMMENT '剩余数量',
  `url` text COLLATE utf8mb4_general_ci COMMENT '兑换链接',
  `super_url` text COLLATE utf8mb4_general_ci COMMENT '超级跳转链接对象[h5,小程序]',
  `isphysical` int NOT NULL COMMENT '奖品大类(0:虚拟,1:实物,2:乐豆,3:谢谢参与,4:公益任务,5:成长值)',
  `third_type_code` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '第三方奖品类型编码(i.e.优惠券编码)',
  `third_type_name` varchar(200) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '第三方奖品类型名称(i.e.优惠券名称)',
  `value` int DEFAULT NULL COMMENT '奖品数值(例如:优惠券面额值)',
  `rule` text COLLATE utf8mb4_general_ci COMMENT '兑换规则',
  `gift_type_order` int DEFAULT NULL COMMENT '奖品类型排序',
  `gift_format` int DEFAULT NULL COMMENT '券形式(1:兑换码,2:直接跳转,3:统一码,4:官网优惠券,5:实物,6:乐豆,7:门店核销券,8:成长值,9:来酷门店核销券,10:龙腾兑换码)',
  `claim_type` int NOT NULL COMMENT '领取形式(1:需要手工领取,2:发放完直接已领取)',
  `preset_type` int NOT NULL DEFAULT '0' COMMENT '预置类型(0:非预置,1:预置)',
  `gift_pool_url` varchar(300) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '奖池文件url',
  `ledou_cost_type` int DEFAULT '0' COMMENT '是否需要消耗乐豆(0:不用,1:需要)',
  `ledou_cost_value` int DEFAULT NULL COMMENT '需要消耗的乐豆数',
  `benefit_type` int DEFAULT '0' COMMENT '权益类型(0:默认普通券,1:生日券,2:新人券)',
  `state` int NOT NULL DEFAULT '1' COMMENT '激活状态(0:未激活,1:已激活)',
  `order_time` datetime DEFAULT NULL COMMENT '预约时间',
  `start_time` datetime NOT NULL COMMENT '开始时间',
  `end_time` datetime NOT NULL COMMENT '结束时间',
  `show_start_time` datetime DEFAULT NULL COMMENT '展示开始时间',
  `show_end_time` datetime DEFAULT NULL COMMENT '展示结束时间',
  `tip_name` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '提示-名称',
  `tip_desc` varchar(500) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '提示-描述',
  `tip_expire` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '提示-有效期',
  `tip_restrict` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '提示-平台限制',
  `activityid` int DEFAULT NULL COMMENT '活动id(实际当做gift_type的分类使用)',
  `actual_activityid` int DEFAULT NULL COMMENT '实际活动id',
  `gift_source` int DEFAULT NULL COMMENT '奖品类型(券)来源',
  `goods_id` int DEFAULT NULL COMMENT '库存商品id',
  `goods_count` int DEFAULT '1' COMMENT '每个奖品包含库存商品的数量',
  `clientid` int NOT NULL DEFAULT '1' COMMENT '客户端(0:全端,1:PC端,2:移动端,3:微信端,4:小程序端,5:APP端)',
  `tenant_id` int NOT NULL COMMENT '租户Id(25:消费线下)',
  `create_time` datetime NOT NULL COMMENT '创建时间',
  `create_by` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '创建人',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `update_by` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '更新人',
  PRIMARY KEY (`code`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=15234 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='奖品类型码表';

