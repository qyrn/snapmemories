import type { Language } from "../../src/ui/strings.ts";

export interface FilmText {
  introTitle: string;
  newTab: string;
  introSubtitle: string;
  stepWord: string;
  requestSent: string;
  mailFrom: string;
  mailWait: string;
  recentDownloads: string;
  downloadDone: string;
  siteTab: string;
  dialogTitle: string;
  dialogPlaces: string[];
  dialogFolders: string[];
  dialogFolderLabel: string;
  dialogSelect: string;
  dialogCancel: string;
  explorerPath: string[];
  infoTaken: string;
  infoPlace: string;
  infoStickers: string;
  outroTitle: string;
  outroSubtitle: string;
}

export const FILM_TEXT: Record<Language, FilmText> = {
  fr: {
    introTitle: "Récupère tes Souvenirs Snapchat",
    newTab: "Nouvel onglet",
    introSubtitle: "Le tuto complet, en 8 étapes",
    stepWord: "Étape",
    requestSent: "Demande envoyée à Snapchat",
    mailFrom: "Snapchat",
    mailWait: "De quelques minutes à quelques heures",
    recentDownloads: "Téléchargements récents",
    downloadDone: "21,7 Mo · Terminé",
    siteTab: "SnapMemories",
    dialogTitle: "Sélectionner un dossier",
    dialogPlaces: ["Accueil", "Bureau", "Téléchargements", "Documents", "Images"],
    dialogFolders: ["Captures d'écran", "Pellicule", "Souvenirs Snapchat"],
    dialogFolderLabel: "Dossier :",
    dialogSelect: "Sélectionner un dossier",
    dialogCancel: "Annuler",
    explorerPath: ["Images", "Souvenirs Snapchat", "2025", "2025-06"],
    infoTaken: "Prise le 1 juin 2025 à 02:11",
    infoPlace: "Lieu : 48,8995° N, 6,0494° E",
    infoStickers: "Stickers recollés sur la photo",
    outroTitle: "memories.qyrn.dev",
    outroSubtitle: "Gratuit · Rien à installer · Rien n'est envoyé",
  },
  en: {
    introTitle: "Get your Snapchat Memories back",
    newTab: "New Tab",
    introSubtitle: "The full tutorial, in 8 steps",
    stepWord: "Step",
    requestSent: "Request sent to Snapchat",
    mailFrom: "Snapchat",
    mailWait: "A few minutes to a few hours",
    recentDownloads: "Recent downloads",
    downloadDone: "21.7 MB · Done",
    siteTab: "SnapMemories",
    dialogTitle: "Select Folder",
    dialogPlaces: ["Home", "Desktop", "Downloads", "Documents", "Pictures"],
    dialogFolders: ["Screenshots", "Camera Roll", "Snapchat Memories"],
    dialogFolderLabel: "Folder:",
    dialogSelect: "Select Folder",
    dialogCancel: "Cancel",
    explorerPath: ["Pictures", "Snapchat Memories", "2025", "2025-06"],
    infoTaken: "Taken on June 1, 2025 at 02:11",
    infoPlace: "Place: 48.8995° N, 6.0494° E",
    infoStickers: "Stickers merged onto the photo",
    outroTitle: "memories.qyrn.dev",
    outroSubtitle: "Free · Nothing to install · Nothing uploaded",
  },
};
