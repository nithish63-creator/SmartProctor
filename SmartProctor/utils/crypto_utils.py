# utils/crypto_utils.py
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import os

def sign_bytes(data: bytes, private_key_pem_path: str, password: bytes = None) -> bytes:
    """
    Sign raw bytes using RSA private key in PEM format.
    Returns the raw signature bytes (not base64).
    """
    if not os.path.exists(private_key_pem_path):
        raise FileNotFoundError(f"Private key not found: {private_key_pem_path}")
    with open(private_key_pem_path, "rb") as f:
        key_data = f.read()
    priv = serialization.load_pem_private_key(key_data, password=password)
    signature = priv.sign(
        data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return signature

def verify_signature_bytes(data: bytes, signature: bytes, public_key_pem_path: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric import padding as _padding
    if not os.path.exists(public_key_pem_path):
        raise FileNotFoundError(f"Public key not found: {public_key_pem_path}")
    with open(public_key_pem_path, "rb") as f:
        pub_data = f.read()
    pub = serialization.load_pem_public_key(pub_data)
    try:
        pub.verify(
            signature,
            data,
            _padding.PSS(mgf=_padding.MGF1(hashes.SHA256()), salt_length=_padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False
