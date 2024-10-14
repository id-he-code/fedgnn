from phe import paillier
import numpy as np

# 生成公钥和私钥
public_key, private_key = paillier.generate_paillier_keypair ()


def encrypt_vector(vector, public_key):
    """加密向量"""
    return [public_key.encrypt (x) for x in vector]


def homomorphic_dot_product(enc_vec1, vec2):
    """在加密域中计算点积"""
    return sum ([enc_vec1[i] * vec2[i] for i in range (len (vec2))])


def homomorphic_norm(enc_vec):
    """在加密域中计算向量的范数（平方和），然后解密"""
    encrypted_square_sum = sum ([x * x for x in enc_vec])
    decrypted_square_sum = private_key.decrypt (encrypted_square_sum)
    return np.sqrt (decrypted_square_sum)


def cosine_similarity_homomorphic(vec1, vec2, public_key, private_key):
    """加密计算余弦相似度"""
    enc_vec1 = encrypt_vector (vec1, public_key)

    # 加密域中的点积计算
    encrypted_dot_product = homomorphic_dot_product (enc_vec1, vec2)
    dot_product = private_key.decrypt (encrypted_dot_product)

    # 向量范数
    norm_vec1 = homomorphic_norm (enc_vec1)
    norm_vec2 = np.linalg.norm (vec2)

    # 计算余弦相似度
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)



