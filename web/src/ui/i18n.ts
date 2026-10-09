export type Language = "fr" | "en";

const STORAGE_KEY = "snapmemories-language";

const STRINGS = {
  en: {
    pageTitle: "SnapMemories: save your Snapchat Memories",
    pageDescription:
      "Get every Snapchat Memory back on your computer with its date and place. Free, nothing to install, nothing leaves your device.",
    badge: "Free · Nothing to install · Stays on your device",
    heroTitle: "Save your Snapchat Memories before they're gone.",
    heroText:
      "Snapchat only keeps 5 GB of Memories for free. Above that, you pay or you export. Get every photo and video back on your computer, with its date, its place and its stickers.",
    howTitle: "Get your export from Snapchat",
    step1Title: "Open the Snapchat download page",
    step1Text: "Sign in with your Snapchat account.",
    step2Title: 'Turn on "Export your Memories" and "Export JSON Files"',
    step2Text: 'Choose "All time" as the date range, then submit.',
    step3Title: "Wait for the email from Snapchat",
    step3Text: "It can take a few minutes or a few hours.",
    step4Title: "Download every ZIP file from the email",
    step4Text: "Big exports come in several parts. Keep them zipped. The links expire after 7 days.",
    dropTitle: "Drop your Snapchat ZIP files",
    dropHint: "or click to choose them. Drop all the parts at once.",
    dropPrivacy: "Nothing is uploaded: your files are read inside this tab.",
    reading: "Reading your export",
    photos: "Photos",
    videos: "Videos",
    size: "Size",
    noteLocated: "With a location: {count}",
    noteMissing:
      "Not found in these files: {count}. If Snapchat sent several ZIP files, drop them all together.",
    noteLinkOnly: "Only available as download links: {count}. Use the Windows app for those.",
    chooseFolder: "Choose where to save",
    chooseFolderHint:
      'Pick or create a folder, for example "Snapchat Memories" in your Pictures. Already saved memories will be skipped.',
    downloadZip: "Save as ZIP files",
    downloadZipHint:
      "Your browser can't write to a folder, so you'll get ZIP files of 1 GB max. Chrome or Edge on a computer can save straight into a folder.",
    startOver: "Start over",
    savingTitle: "Saving your memories",
    savingText: "Keep this tab open until the end.",
    saved: "Saved",
    failed: "Failed",
    remaining: "Remaining",
    stop: "Stop",
    stopping: "Stopping",
    doneTitle: "Memories saved!",
    doneText: "Everything is sorted by year and month.",
    doneIncompleteTitle: "Almost everything saved",
    stoppedTitle: "Import stopped",
    stoppedText: "Drop the same files again later: saved memories will be skipped.",
    alreadyThere: "Already there",
    partsTitle: "Your ZIP files",
    partsText: "Downloads should start by themselves. If not, click each file.",
    errorsTitle: "Not saved",
    importAnother: "Import another export",
    progressLabel: "{done} / {total} memories",
    skippedLabel: "Already saved, skipped: {count}.",
    leaveWarning: "Your memories are still being saved.",
    errorInvalidZip: "{name} is not a ZIP file from Snapchat.",
    errorNoMemories: 'No memories found. Check that "Export your Memories" was turned on.',
    errorInvalidHistory: "The memories list in this export is damaged.",
    errorNothingToImport: "No photos or videos found in these files.",
    errorFolder: "This folder can't be used. Choose another one.",
    errorUnexpected: "Something went wrong. Reload the page and try again.",
    faqTitle: "Questions",
    faqSafeTitle: "Is it safe?",
    faqSafeText:
      "Your export never leaves your device. This page has no server behind it: it reads the ZIP files inside your browser and writes the result to your disk. The code is open source.",
    faqWhatTitle: "What do I get?",
    faqWhatText:
      "Your photos and videos sorted by year and month, with the real date and place written inside each file. Google Photos, Apple Photos and Windows show them at the right date.",
    faqPhoneTitle: "Can I do it on my phone?",
    faqPhoneText: "It works for small exports. For a big library, use a computer with Chrome or Edge.",
    faqDesktopTitle: "Is there an app?",
    faqDesktopText: "Yes, for Windows. It's useful for very old exports that only contain download links.",
    desktopLink: "Download the Windows app",
    footerSource: "Open source on GitHub",
    footerMadeBy: "Made by qyrn",
    language: "Français",
  },
  fr: {
    pageTitle: "SnapMemories : récupère tes Memories Snapchat",
    pageDescription:
      "Récupère toutes tes Memories Snapchat sur ton ordinateur avec leur date et leur lieu. Gratuit, rien à installer, rien ne quitte ton appareil.",
    badge: "Gratuit · Rien à installer · Tout reste sur ton appareil",
    heroTitle: "Récupère tes Memories Snapchat avant de les perdre.",
    heroText:
      "Snapchat ne garde plus que 5 Go de Memories gratuitement. Au-delà, tu paies ou tu exportes. Récupère chaque photo et vidéo sur ton ordinateur, avec sa date, son lieu et ses stickers.",
    howTitle: "Demande ton export à Snapchat",
    step1Title: "Ouvre la page de téléchargement Snapchat",
    step1Text: "Connecte-toi avec ton compte Snapchat.",
    step2Title: 'Active "Export your Memories" et "Export JSON Files"',
    step2Text: 'Choisis "All time" comme période, puis valide.',
    step3Title: "Attends le mail de Snapchat",
    step3Text: "Ça peut prendre quelques minutes ou quelques heures.",
    step4Title: "Télécharge tous les ZIP du mail",
    step4Text:
      "Les gros exports arrivent en plusieurs parties. Ne les décompresse pas. Les liens expirent au bout de 7 jours.",
    dropTitle: "Dépose tes ZIP Snapchat",
    dropHint: "ou clique pour les choisir. Dépose toutes les parties d'un coup.",
    dropPrivacy: "Rien n'est envoyé : tes fichiers sont lus dans cet onglet.",
    reading: "Lecture de ton export",
    photos: "Photos",
    videos: "Vidéos",
    size: "Taille",
    noteLocated: "Avec un lieu : {count}",
    noteMissing:
      "Introuvables dans ces fichiers : {count}. Si Snapchat t'a envoyé plusieurs ZIP, dépose-les tous ensemble.",
    noteLinkOnly: "Disponibles seulement sous forme de lien : {count}. Utilise l'app Windows pour celles-là.",
    chooseFolder: "Choisir où enregistrer",
    chooseFolderHint:
      'Choisis ou crée un dossier, par exemple "Memories Snapchat" dans tes Images. Les memories déjà enregistrées seront ignorées.',
    downloadZip: "Enregistrer en fichiers ZIP",
    downloadZipHint:
      "Ton navigateur ne peut pas écrire dans un dossier : tu recevras des ZIP de 1 Go maximum. Chrome ou Edge sur ordinateur enregistrent directement dans un dossier.",
    startOver: "Recommencer",
    savingTitle: "Enregistrement en cours",
    savingText: "Garde cet onglet ouvert jusqu'à la fin.",
    saved: "Enregistrées",
    failed: "Échecs",
    remaining: "Restant",
    stop: "Arrêter",
    stopping: "Arrêt en cours",
    doneTitle: "Memories enregistrées !",
    doneText: "Tout est rangé par année et par mois.",
    doneIncompleteTitle: "Presque tout est enregistré",
    stoppedTitle: "Import arrêté",
    stoppedText: "Redépose les mêmes fichiers plus tard : les memories déjà enregistrées seront ignorées.",
    alreadyThere: "Déjà là",
    partsTitle: "Tes fichiers ZIP",
    partsText: "Les téléchargements démarrent tout seuls. Sinon, clique sur chaque fichier.",
    errorsTitle: "Non enregistrées",
    importAnother: "Importer un autre export",
    progressLabel: "{done} / {total} memories",
    skippedLabel: "Déjà enregistrées, ignorées : {count}.",
    leaveWarning: "Tes memories sont encore en cours d'enregistrement.",
    errorInvalidZip: "{name} n'est pas un ZIP de Snapchat.",
    errorNoMemories: 'Aucune memory trouvée. Vérifie que "Export your Memories" était activé.',
    errorInvalidHistory: "La liste des memories de cet export est abîmée.",
    errorNothingToImport: "Aucune photo ni vidéo dans ces fichiers.",
    errorFolder: "Ce dossier ne peut pas être utilisé. Choisis-en un autre.",
    errorUnexpected: "Un problème est survenu. Recharge la page et réessaie.",
    faqTitle: "Questions",
    faqSafeTitle: "C'est sûr ?",
    faqSafeText:
      "Ton export ne quitte jamais ton appareil. Cette page n'a aucun serveur derrière : elle lit les ZIP dans ton navigateur et écrit le résultat sur ton disque. Le code est open source.",
    faqWhatTitle: "Qu'est-ce que je récupère ?",
    faqWhatText:
      "Tes photos et vidéos rangées par année et par mois, avec la vraie date et le lieu écrits dans chaque fichier. Google Photos, Apple Photos et Windows les affichent à la bonne date.",
    faqPhoneTitle: "Je peux le faire sur mon téléphone ?",
    faqPhoneText:
      "Oui pour les petits exports. Pour une grosse bibliothèque, utilise un ordinateur avec Chrome ou Edge.",
    faqDesktopTitle: "Il existe une app ?",
    faqDesktopText:
      "Oui, pour Windows. Elle sert pour les très vieux exports qui ne contiennent que des liens de téléchargement.",
    desktopLink: "Télécharger l'app Windows",
    footerSource: "Open source sur GitHub",
    footerMadeBy: "Fait par qyrn",
    language: "English",
  },
} as const satisfies Record<Language, Record<string, string>>;

export type MessageKey = keyof (typeof STRINGS)["en"];

function storedLanguage(): Language | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === "fr" || value === "en" ? value : null;
  } catch {
    return null;
  }
}

let current: Language = storedLanguage() ?? (navigator.language.toLowerCase().startsWith("fr") ? "fr" : "en");

export function language(): Language {
  return current;
}

export function setLanguage(next: Language): void {
  current = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    return;
  }
}

export function t(key: MessageKey, values: Record<string, string | number> = {}): string {
  return STRINGS[current][key].replace(/\{(\w+)\}/g, (_, name: string) => String(values[name] ?? ""));
}

export function isMessageKey(value: string): value is MessageKey {
  return value in STRINGS.en;
}
