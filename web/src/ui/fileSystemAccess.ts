interface DirectoryPickerOptions {
  id?: string;
  mode?: "read" | "readwrite";
  startIn?: "desktop" | "documents" | "downloads" | "pictures";
}

type DirectoryPicker = (options?: DirectoryPickerOptions) => Promise<FileSystemDirectoryHandle>;

function picker(): DirectoryPicker | null {
  const candidate: unknown = Reflect.get(window, "showDirectoryPicker");
  return typeof candidate === "function" ? (candidate as DirectoryPicker).bind(window) : null;
}

export function canWriteFolders(): boolean {
  return picker() !== null;
}

export async function pickFolder(): Promise<FileSystemDirectoryHandle | null> {
  const show = picker();
  if (!show) return null;
  try {
    return await show({ id: "snapmemories", mode: "readwrite", startIn: "pictures" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    throw error;
  }
}
