import cv2  # Bibliothèque pour le traitement d'image (OpenCV)
import numpy as np  # Bibliothèque pour les calculs matriciels
import tkinter as tk  # Bibliothèque standard pour l'interface graphique (GUI)
from tkinter import filedialog, messagebox  # Modules pour ouvrir des fichiers et afficher des alertes
from PIL import Image, ImageTk  # Utilisé pour convertir les images OpenCV vers un format compatible Tkinter
from pyzbar.pyzbar import decode  # Bibliothèque spécialisée dans le décodage des codes-barres et QR
import os  # Pour les opérations liées au système de fichiers

class ScannerApp:
    def __init__(self, root):
        self.root = root  # Initialisation de la fenêtre principale
        self.root.title("Scanner Professionnel de QR Code & Code-barres")  # Titre de la fenêtre
        self.root.geometry("1000x700")  # Définition de la taille par défaut
        
        # --- Variables de stockage ---
        self.current_img = None  # Variable pour stocker l'image originale chargée
        self.annotated_img = None  # Variable pour stocker l'image après détection (avec les cadres)

        # --- Éléments de l'interface ---
        self.setup_ui()  # Appel de la fonction de création des boutons et zones d'affichage

    def setup_ui(self):
        # Création du panneau latéral gauche (couleur sombre #2c3e50)
        self.ctrl_panel = tk.Frame(self.root, width=250, bg="#2c3e50")
        self.ctrl_panel.pack(side=tk.LEFT, fill=tk.Y)  # Placé à gauche, remplit toute la hauteur

        # Bouton pour charger une image
        self.btn_load = tk.Button(self.ctrl_panel, text="Charger Image", command=self.load_image_gui, 
                                 bg="#3498db", fg="white", font=("Arial", 10, "bold"), pady=10)
        self.btn_load.pack(padx=10, pady=20, fill=tk.X)  # Marge interne et remplit la largeur du panneau

        # Bouton pour lancer le scan
        self.btn_scan = tk.Button(self.ctrl_panel, text="Scanner", command=self.run_pipeline_gui, 
                                 bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), pady=10)
        self.btn_scan.pack(padx=10, pady=10, fill=tk.X)

        # Étiquette de texte pour le titre des résultats
        self.res_label = tk.Label(self.ctrl_panel, text="Résultats détaillés :", bg="#2c3e50", fg="white", font=("Arial", 10))
        self.res_label.pack(pady=(20, 0))

        # Zone de texte défilante pour afficher les données décodées
        self.text_area = tk.Text(self.ctrl_panel, height=15, width=30, font=("Consolas", 9), bg="#ecf0f1")
        self.text_area.pack(padx=10, pady=5)

        # Zone d'affichage principale (à droite) pour l'image
        self.display_panel = tk.Label(self.root, text="Aucune image chargée", bg="#ecf0f1")
        self.display_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    # --- Fonctions de traitement d'image ---

    def preprocess_image(self, img):
        # Conversion de l'image couleur (BGR) en nuances de gris (Gray)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarisation d'Otsu : sépare le noir et le blanc de façon optimale
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Application d'un flou Gaussien pour réduire le bruit numérique
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Algorithme Canny pour détecter les contours (lignes de contraste)
        edges = cv2.Canny(blur, 50, 150)
        
        # Dilatation et Érosion : épaissit les lignes pour boucher les trous dans les contours
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Calcul du gradient de Sobel : accentue les zones de fortes variations (comme les barres d'un code)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
        gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y)) # Soustraction pour isoler les motifs verticaux

        # Flou sur le gradient pour regrouper les barres du code-barres en un bloc
        blurred_grad = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, grad_thresh = cv2.threshold(blurred_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Opération "Closing" : fusionne les barres proches pour créer une zone rectangulaire pleine
        kernel_bar = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(grad_thresh, cv2.MORPH_CLOSE, kernel_bar)
        closed = cv2.erode(closed, None, iterations=4) # Nettoyage des petits bruits
        closed = cv2.dilate(closed, None, iterations=4) # Rétablissement de la taille après érosion

        return {"gray": gray, "thresh": thresh, "edges": edges, "closed": closed}

    def detect_qr_local(self, img, preprocessed):
        qr_detector = cv2.QRCodeDetector() # Initialisation du moteur de détection QR d'OpenCV
        results = []
        edges = preprocessed["edges"]
        
        # Recherche des contours dans l'image traitée par Canny
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # On ignore les zones trop petites pour être un QR Code
            if cv2.contourArea(cnt) < 1000: continue
            
            # On calcule le rectangle englobant le contour
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)
            
            # Un QR Code est toujours approximativement carré (ratio proche de 1)
            if not (0.8 < ratio < 1.2): continue

            # On découpe cette zone (ROI = Region of Interest) de l'image
            roi = img[y:y + h, x:x + w]
            if roi.size == 0: continue
            
            # Prétraitement spécifique de la petite zone pour faciliter la lecture
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Ajout d'une marge blanche (Quiet Zone) indispensable pour les lecteurs de codes
            padding = int(w * 0.15)
            roi_padded = cv2.copyMakeBorder(roi_thresh, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
            
            # Agrandissement pour que les petits QR soient lisibles
            roi_big = cv2.resize(roi_padded, (400, 400), interpolation=cv2.INTER_NEAREST)

            # Test de décodage sur la version originale et la version agrandie
            for candidate in [roi_padded, roi_big]:
                data_local, _, _ = qr_detector.detectAndDecode(candidate)
                if data_local:
                    res_str = f"[QR] {data_local}"
                    if res_str not in results:
                        results.append(res_str)
                        # Dessin d'un rectangle vert si succès
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    break
        return results, img

    def detect_barcodes(self, img, preprocessed):
        results = []
        closed = preprocessed["closed"]
        # Recherche des blocs rectangulaires potentiels pour les codes-barres 1D
        bar_contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in bar_contours:
            # On ignore les blocs trop petits
            if cv2.contourArea(c) < 3000: continue
            
            # Calcul du rectangle incliné pour suivre l'angle du code-barres
            rect = cv2.minAreaRect(c)
            box = np.int32(cv2.boxPoints(rect)) # Coordonnées des 4 coins
            
            # Découpe de la zone
            x, y, w, h = cv2.boundingRect(box)
            roi = img[max(0,y):y+h, max(0,x):x+w]
            if roi.size == 0: continue

            # Amélioration du contraste pour les codes-barres (Égalisation d'histogramme)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_eq = cv2.equalizeHist(roi_gray)
            _, roi_thresh = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_big = cv2.resize(roi_thresh, None, fx=2, fy=2)

            # Test du moteur PyZbar sur plusieurs versions de la zone
            candidates = [roi, roi_gray, roi_eq, roi_thresh, roi_big]
            decoded = []
            for cand in candidates:
                decoded = decode(cand)
                if decoded: break

            for barcode in decoded:
                # On ignore les QR ici car ils sont gérés par la fonction précédente
                if barcode.type == "QRCODE": continue
                barcode_data = barcode.data.decode("utf-8")
                results.append(f"[{barcode.type}] {barcode_data}")
                # Dessin du contour en rouge pour les codes-barres
                cv2.drawContours(img, [box], -1, (0, 0, 255), 3)

        return results, img

    # --- Gestion de l'Interface Graphique ---

    def load_image_gui(self):
       
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.current_img = cv2.imread(path) # Lecture de l'image par OpenCV
            self.display_image(self.current_img) # Mise à jour de l'affichage
            self.text_area.delete('1.0', tk.END) # Effacement des anciens textes

    def run_pipeline_gui(self):
        # Vérification si une image existe
        if self.current_img is None:
            messagebox.showwarning("Attention", "Veuillez charger une image d'abord.")
            return

        annotated = self.current_img.copy() # Copie pour ne pas abîmer l'original
        pre = self.preprocess_image(annotated) # Étape 1 : Prétraitement
        
        # Étape 2 : Détections
        qr_res, annotated = self.detect_qr_local(annotated, pre)
        bar_res, annotated = self.detect_barcodes(annotated, pre)
        
        all_res = qr_res + bar_res
        
        # Mise à jour de la zone de texte à gauche
        self.text_area.delete('1.0', tk.END)
        if all_res:
            for r in all_res:
                self.text_area.insert(tk.END, f"• {r}\n")
        else:
            self.text_area.insert(tk.END, "Aucun code trouvé.")
        
        # Mise à jour de l'image avec les cadres de couleurs
        self.display_image(annotated)

    def display_image(self, img):
        # OpenCV travaille en BGR, mais Tkinter/PIL ont besoin de RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Redimensionnement proportionnel pour que l'image tienne dans la fenêtre
        max_size = (800, 600)
        img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Conversion finale pour l'affichage Tkinter
        img_tk = ImageTk.PhotoImage(img_pil)
        self.display_panel.config(image=img_tk, text="") # Suppression du texte d'attente
        self.display_panel.image = img_tk # Référence indispensable pour éviter que l'image disparaisse

if __name__ == "__main__":
    root = tk.Tk() # Création du moteur d'interface
    app = ScannerApp(root) # Lancement de notre application
    root.mainloop() # Boucle infinie pour maintenir la fenêtre ouverte