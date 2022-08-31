# How to run Streamlit GUI
1. In folder `Fast_text2StyleGAN/`, run: 
   ```bash
   streamlit run ./streamlit_gui/app.py --server.port=8877
   ```
2. On local machine, do:
   ```bash
   ssh -N -L 8877:XXX.XXX.X.XX:8877 username@your.remote-server.com
   ``` 
   make sure to replace XXX.XXX.X.XX with your remote server's ip address
3. Open a web browser on local machine and type: 
   ```bash
   localhost:8877
   ```