"use client";

import { useEffect, useMemo, useState } from 'react';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useToast } from '@/hooks/useToast';
import { useEditorStore } from '@/stores/useEditorStore';
import { useCanvasStore } from '@/stores/useCanvasStore';
import { useHistoryStore } from '@/stores/useHistoryStore';
import type { CanvasImage } from '@/types/canvas';
import { CanvasContainer } from './Canvas/CanvasContainer';
import { PromptComposerProvider, usePromptComposer } from './Composer/PromptComposerProvider';
import { BottomComposer } from './Composer/BottomComposer';
import { ExportModal } from './Export/ExportModal';
import { HistorySidebar } from './Sidebar/HistorySidebar';
import { LeftPanel } from './Sidebar/LeftPanel';
import { TopBar } from './Shell/TopBar';
import { ToastContainer } from './ToastContainer';

type LoadedImageSize = { width: number; height: number };

function loadImageSize(url: string): Promise<LoadedImageSize> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth > 0 && image.naturalHeight > 0) {
        resolve({ width: image.naturalWidth, height: image.naturalHeight });
        return;
      }
      reject(new Error('Loaded project image has no readable dimensions'));
    };
    image.onerror = () => reject(new Error('Project image dimensions could not be loaded'));
    image.src = url;
  });
}

export function EditorLayout() {
  return (
    <PromptComposerProvider>
      <StandaloneEditorShell />
    </PromptComposerProvider>
  );
}

function StandaloneEditorShell() {
  useKeyboardShortcuts();

  const { toasts, addToast, removeToast } = useToast();
  const baseImage = useEditorStore((s) => s.baseImage);
  const setBaseImage = useEditorStore((s) => s.setBaseImage);
  const canvasImageCount = useCanvasStore((s) => s.imageOrder.length);
  const focusedImageIds = useCanvasStore((s) => s.focusedImageIds);
  const hydrateCanvas = useCanvasStore((s) => s.hydrate);
  const hydrateEntries = useHistoryStore((s) => s.hydrateEntries);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [projectName, setProjectName] = useState('Untitled design');

  useEffect(() => {
    let cancelled = false;
    async function hydrateProjectHistory() {
      try {
        const sessionRes = await fetch('/api/projects/current', { cache: 'no-store' });
        if (!sessionRes.ok) {
          const payload = await sessionRes.json().catch(() => null) as { error?: string } | null;
          throw new Error(payload?.error || 'Project session could not be loaded');
        }
        const session = await sessionRes.json();
        if (session.persistence !== 'project') return;
        if (typeof session.projectName === 'string' && session.projectName.trim()) {
          setProjectName(session.projectName);
        }
        const historyRes = await fetch('/api/projects/history', { cache: 'no-store' });
        if (!historyRes.ok) {
          const payload = await historyRes.json().catch(() => null) as { error?: string } | null;
          throw new Error(payload?.error || 'Project history could not be loaded');
        }
        const history = await historyRes.json();
        if (cancelled || !Array.isArray(history.entries)) return;
        const first = history.entries[0];
        const rootImageId = typeof first?.imageId === 'string' ? first.imageId : (typeof first?.id === 'string' ? first.id : null);
        const entries = history.entries.map((entry: { id?: string; imageId?: string; assetUrl?: string; imageDataUrl?: string }) => ({
          ...entry,
          imageId: entry.imageId ?? rootImageId ?? undefined,
          imageDataUrl: entry.imageDataUrl ?? entry.assetUrl,
        }));
        hydrateEntries(entries);
        if (first?.assetUrl) {
          const imageSize = await loadImageSize(first.assetUrl);
          if (cancelled) return;
          setBaseImage(first.assetUrl, imageSize);
          if (rootImageId && canvasImageCount === 0) {
            const rootImage: CanvasImage = {
              id: rootImageId,
              url: first.assetUrl,
              assetId: typeof first.assetId === 'string' ? first.assetId : undefined,
              size: imageSize,
              position: { x: 0, y: 0 },
              parentId: null,
              generationIndex: 0,
              prompt: typeof first.prompt === 'string' ? first.prompt : '',
              provider: first.provider === 'god-tibo' ? 'god-tibo' : 'openai',
              type: first.type === 'edit' ? 'edit' : 'generate',
              createdAt: typeof first.timestamp === 'number' ? first.timestamp : Date.now(),
              paths: [],
              boxes: [],
              memos: [],
              status: 'ready',
            };
            hydrateCanvas({ [rootImageId]: rootImage }, [rootImageId], [rootImageId]);
          }
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Project history could not be loaded';
          addToast(message, 'error', { durationMs: 8000 });
        }
      }
    }
    void hydrateProjectHistory();
    return () => { cancelled = true; };
  }, [addToast, canvasImageCount, hydrateCanvas, hydrateEntries, setBaseImage]);

  useEffect(() => {
    if (!baseImage || canvasImageCount > 0) return;
    const editor = useEditorStore.getState();
    const id = crypto.randomUUID();
    const rootImage: CanvasImage = {
      id,
      url: baseImage,
      size: editor.imageSize.width > 0 && editor.imageSize.height > 0
        ? editor.imageSize
        : { width: 1024, height: 1024 },
      position: { x: 0, y: 0 },
      parentId: null,
      generationIndex: 0,
      prompt: '',
      provider: 'openai',
      type: 'generate',
      createdAt: Date.now(),
      paths: editor.paths,
      boxes: editor.boxes,
      memos: editor.memos,
      status: 'ready',
    };
    hydrateCanvas({ [id]: rootImage }, [id], [id]);
  }, [baseImage, canvasImageCount, hydrateCanvas]);

  const {
    prompt,
    setPrompt,
    referenceImages,
    systemPrompt,
    setSystemPrompt,
    designContext,
    designContextFileName,
    replaceDesignContext,
    clearDesignContext,
    addReferenceFiles,
    removeReferenceImage,
    clearReferenceImages,
    handleGenerate,
    handleEdit,
  } = usePromptComposer();

  const referencePreviews = useMemo(() => (
    referenceImages.map((reference) => ({
      id: reference.id,
      previewUrl: reference.previewUrl,
      file: reference.file,
      name: reference.file.name,
    }))
  ), [referenceImages]);

  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-[#1e1e1e] text-[#e6e6e6]">
      <TopBar canExport={focusedImageIds.length > 0} onExportClick={() => setIsExportOpen(true)} projectName={projectName} />
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        <LeftPanel
          references={referencePreviews}
          onAddReferenceFiles={addReferenceFiles}
          onRemoveReference={removeReferenceImage}
          onClearReferences={clearReferenceImages}
          systemPrompt={systemPrompt}
          onSystemPromptChange={setSystemPrompt}
          designContext={designContext}
          designContextFileName={designContextFileName}
          onReplaceDesignContext={replaceDesignContext}
          onClearDesignContext={clearDesignContext}
        />
        <main className="relative min-w-0 flex-1 overflow-hidden">
          <CanvasContainer className="h-full w-full" />
        </main>
        <HistorySidebar />
      </div>
      <BottomComposer
        prompt={prompt}
        onPromptChange={setPrompt}
        references={referencePreviews}
        onAddReferenceFiles={addReferenceFiles}
        onRemoveReference={removeReferenceImage}
        onGenerate={handleGenerate}
        onEdit={handleEdit}
      />
      <ExportModal open={isExportOpen} onOpenChange={setIsExportOpen} canExport={focusedImageIds.length > 0} />
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}
