import os
import yt_dlp
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# Diretório dos dados
OUTPUT_DIR = "../data/raw"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def download_audio_ytdlp(url, output_dir=OUTPUT_DIR):
    """
    Baixa o áudio de um vídeo do YouTube utilizando yt-dlp e converte para WAV.

    Args:
        url (str): URL do vídeo do YouTube.
        output_dir (str, optional): Caminho absoluto do diretório onde o áudio será salvo.
                                    Default: "data/raw".

    Returns:
        str: Caminho do arquivo de áudio salvo, ou None em caso de erro.
    """
    # Converte para caminho absoluto
    output_dir = os.path.abspath(output_dir)

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("O diretório de saída fornecido não é válido.")

    ytdlp_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl':os.path.join(output_dir, '%(title)s.%(ext)s')
    }

    try:
        with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
            # Garante que o nome do arquivo final esteja correto
            final_path = filename.replace('.webm', '.wav').replace('.m4a', '.wav')
            return final_path  # Retorna o caminho correto do arquivo baixado
    except Exception as e:
        print(f"Erro na execução do yt-dlp: {e}")
        return None

def extract_segments(audio_path, output_dir=OUTPUT_DIR):
    """Extrai segmentos de 10 segundos a cada 1 minuto.

    Args:
        audio_path (str): Path do áudio do vídeo baixado
        output_dir (str, optional): Caminho de saída para o arquivo de áudio. Por padrão, OUTPUT_DIR.

    Returns:
        list: Lista contendo os paths dos segmentos de áudio
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        video_id = Path(audio_path).stem
        duration = len(audio) # Duração total em ms

        segment_paths = []
        for start_time in range(10000, duration, 60000): # 0:10, 1:10, 2:10...
            segment = audio[start_time: start_time + 10000]
            segment_filename = f"{video_id}_{start_time//1000}s.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            segment.export(segment_path, format="wav")
            segment_paths.append(segment_path)

        os.remove(audio_path) # Remove o arquivo original após processamento
        return segment_paths

    except Exception as e:
        return f"Error ao processar {audio_path}: {e}"

def process_video(url):
    """Pipeline completo: Baixar áudio e extrair segmentos.

    Args:
        url (str): URL do áudio do vídeo a ser baixado

    Returns:
        list: Lista contendo os paths dos segmentos de áudio
    """
    audio_path = download_audio_ytdlp(url)
    if os.path.exists(audio_path):
        return extract_segments(audio_path)
    return f"Erro no download de {url}"

def process_urls(url_list):
    """Processa múltiplos vídeos em paralelo.

    Args:
        url_list (list): Lista de URLs onde se deseja obter as amostras

    Returns:
        list: Lista contendo os paths dos segmentos de áudio
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_video, url): url for url in url_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(url_list), desc="Processando vídeos"):
            results.append(future.result())
    return results