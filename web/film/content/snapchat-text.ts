import type { Language } from "../../src/ui/strings.ts";

export interface SnapchatText {
  pageTitle: string;
  nav: string[];
  search: string;
  download: string;
  menu: string[];
  heading: string;
  banner: string;
  readyTitle: string;
  intro: string;
  paragraph: string;
  limit: string;
  exportsTitle: string;
  exportsCreated: string;
  exportsAvailable: string;
  seeExports: string;
  hideExports: string;
  exportFile: string;
  downloadButton: string;
  selectTitle: string;
  selectedNone: string;
  selectedOne: string;
  rows: Array<[title: string, subtitle: string]>;
  panelTitle: string;
  panelText: string;
  requestButton: string;
}

export const SNAPCHAT_TEXT: Record<Language, SnapchatText> = {
  fr: {
    pageTitle: "Télécharger mes données | Snapchat",
    nav: ["Stories", "Spotlight", "Chatter", "Lenses", "Snapchat+"],
    search: "Rechercher",
    download: "Télécharger",
    menu: [
      "Mes données",
      "Mon Snapcode",
      "Snapchat+",
      "Visibilité du contenu sur le web",
      "Gestion des contacts",
      "Mes signalements",
      "Ads Manager",
      "Gérer les applications",
      "Gestion des sessions",
    ],
    heading: "Mes données",
    banner:
      "Peur de perdre vos Souvenirs ? Ils ne seront pas supprimés automatiquement si vous choisissez de ne pas passer à la formule supérieure. Vous conserverez l'accès à vos Souvenirs de l'année écoulée ainsi qu'à vos 5 Go de Souvenirs les plus anciens.",
    readyTitle: "Données prêtes à être téléchargées",
    intro: "Votre compte, vos données.",
    paragraph:
      "Exportez une copie du contenu de votre compte Snapchat pour le sauvegarder. Vos sélections et la quantité de données dont nous disposons à votre sujet peuvent avoir une incidence sur le temps nécessaire à la préparation de vos données en vue de leur téléchargement.",
    limit: "Remarque : les demandes de données sont limitées à une toutes les 72 heures.",
    exportsTitle: "Vos exports",
    exportsCreated: "Données créées à 9 oct., 03:51",
    exportsAvailable: "1 export disponible(s). Expiration : 2d 23h 59m",
    seeExports: "Voir les exports",
    hideExports: "Masquer les exportations",
    exportFile: "mydata~1791510669777.zip, 21.7 MiB",
    downloadButton: "Télécharger",
    selectTitle: "Sélectionner les données à inclure",
    selectedNone: "0 / 10 sélectionnés",
    selectedOne: "1 / 10 sélectionnés",
    rows: [
      ["Exporter mes souvenirs", ""],
      ["Exporter des fichiers JSON", "À des fins de portabilité des données"],
      [
        "Données utilisateur",
        "Identifiants, Snapchat+, profil utilisateur, profil public, historique du compte",
      ],
      ["Historique des Chats", "Historique des Snaps, historique des chats, historique des conversations"],
      ["Spotlight", "Story partagée et Spotlight, réponses Spotlight, historique de la Story"],
    ],
    panelTitle: "Vous recherchez simplement un moyen simple de récupérer vos souvenirs ?",
    panelText: "Cliquez ci-dessous pour les exporter maintenant.",
    requestButton: "Demander uniquement des souvenirs",
  },
  en: {
    pageTitle: "Download My Data | Snapchat",
    nav: ["Stories", "Spotlight", "Chat", "Lenses", "Snapchat+"],
    search: "Search",
    download: "Download",
    menu: [
      "My Data",
      "My Snapcode",
      "Snapchat+",
      "Content Visibility on Web",
      "Contact Management",
      "My Reports",
      "Ads Manager",
      "Manage Apps",
      "Session Management",
    ],
    heading: "My Data",
    banner:
      "Worried about losing your Memories? We won't automatically delete them if you choose not to upgrade. You'll continue to have access to Memories saved within the last year and your oldest 5 GB of Memories.",
    readyTitle: "Data available for download",
    intro: "Your account, your data.",
    paragraph:
      "Export a copy of the content in your Snapchat Account to back it up. Your selections and how much data we have about you may impact the amount of time it takes to prepare your data for download.",
    limit: "Please note: Data requests are limited to one request every 72 hours.",
    exportsTitle: "Your exports",
    exportsCreated: "Data created at Oct 9, 03:51 AM",
    exportsAvailable: "1 export available. Expire in 2d 23h 59m",
    seeExports: "See exports",
    hideExports: "Hide exports",
    exportFile: "mydata~1791510669777.zip, 21.7 MiB",
    downloadButton: "Download",
    selectTitle: "Select data to include",
    selectedNone: "0 / 10 selected",
    selectedOne: "1 / 10 selected",
    rows: [
      ["Export your Memories", ""],
      ["Export JSON Files", "For data portability purposes"],
      ["User Information", "Login, Snapchat+, User Profile, Public Profile, Account History"],
      ["Chat History", "Snap History, Chat History, Talk History, Communities"],
      ["Spotlight", "Shared Story & Spotlight, Spotlight Replies, Story History"],
    ],
    panelTitle: "Just Looking for an Easy Way to Get Your Memories?",
    panelText: "Click below to export them now.",
    requestButton: "Request Only Memories",
  },
};
