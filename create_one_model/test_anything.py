from k_model import *
args = ModelConfig(
    dim=1024,
    n_layers=18,
)

# %%
# test RMSNorm
# from k_model import RMSNorm
# def test_rmsnorm():
#     """测试RMSNorm"""
#     x = torch.randn(1, 50, args.dim)
#     rmsnorm = RMSNorm(args.dim, args.norm_eps)
#     print("Input shape:", x.shape)
#     output = rmsnorm(x)
#     print("RMSNorm output shape:", output.shape)
#     assert output.shape == x.shape, "RMSNorm output shape mismatch"
# test_rmsnorm()

def test_repeat_kv():
    """测试repeat_kv"""
    # 这里的输入x的形状是4维的 批量大小为1，序列长度为50，键/值对头的数量，每个头的维度
    model_parallel_size = 1
    x = torch.randn(1, 50, args.n_heads  // model_parallel_size,args.dim // args.n_heads)  # 输入张量
    kv = torch.randn(1, 50, args.n_heads // model_parallel_size, args.dim // args.n_heads)  # 键/值对张量
    output = repeat_kv(x, kv)
    print("Input shape:", x.shape)
    print("KV shape:", kv.shape)
    print("Output shape:", output.shape)
    # assert output.shape == (1, 50, args.dim * 2), "repeat_kv output shape mismatch"
# test_repeat_kv()

def test_apply_rotary_emb():
    """测试apply_rotary_emb"""
    # x = torch.randn(1, 50, args.dim)  # 输入张量
    xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
    print("Input shape:", xq.shape)
    cos, sin = precompute_freqs_cis(288//6, 50)  # 预计算的频率和相位信息
    print("cos sin shape:", cos.shape, sin.shape)

    xq_rot, xk_rot = apply_rotary_emb(xq, xk, cos, sin)
    print("Rotary Embedding shapes:", xq_rot.shape, xk_rot.shape)
    assert xq_rot.shape == xq.shape, "apply_rotary_emb xq shape mismatch"

# test_apply_rotary_emb()

def test_attention():
    """测试LLaMAAttention"""
    # 模拟输入
    x = torch.randn(1, 50, args.dim)  # 输入张量
    kv = torch.randn(1, 50, args.dim)  # 键/值对张量
    cos, sin = precompute_freqs_cis(args.dim//args.n_heads, 50)  # 预计算的频率和相位信息  
    Attention_model = Attention(args)
    
    print("Input shape:", x.shape)
    output = Attention_model(x, cos,sin)
    print("Attention output shape:", output.shape)
    assert output.shape == x.shape, "LLaMAAttention output shape mismatch"
# test_attention()

# 测试MLP模块
def test_MLP():
    """测试MLP"""
    x = torch.randn(1, 50, args.dim)  # 输入张量
    print("Input shape:", x.shape)
    mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    output = mlp(x)
    print("MLP output shape:", output.shape)
    assert output.shape == x.shape, "MLP output shape mismatch"
# test_MLP()

def test_DecoderLayer():
    """测试DecoderLayer"""
    seq_len = 50
    x = torch.randn(1, seq_len, args.dim)  # 输入张量
    print("Input shape:", x.shape)
    decoderlayer = DecoderLayer(0,args) #第一个参数定义的是DecoderLayer所在的层数
    cos, sin  = precompute_freqs_cis(args.dim//args.n_heads, seq_len)
    output = decoderlayer(x, cos, sin)
    print("DecoderLayer output shape:", output.shape)
    assert output.shape == x.shape, "DecoderLayer output shape mismatch"
# test_DecoderLayer()

def test_transformer():
    """测试Transformer"""  
    # 文本对文本生成模型
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_k")
    model = Transformer(args=args)
    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    # input_id = tokenizer(text).data['input_ids']
    input_id = tokenizer.encode(text)  # 推荐写法
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))

    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)
    # 传入模型
    output = model(X,Y)
    # print("Output shape:", output.shape)
    print("#######################################")

    #可以将token_id传入
    input_tensor = torch.tensor(input_id).unsqueeze(0)
    output_ids = model.generate(input_tensor, max_new_tokens=50)
    # output_ids = model.generate(input_id, max_new_tokens=50)
    print("Output ids:", output_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Output text:", output_text)

test_transformer()