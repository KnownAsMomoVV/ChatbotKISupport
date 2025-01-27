import { AccessibilityNode, TreeResult } from "../../types/context";
import { StagehandPage } from "../StagehandPage";
import { LogLine } from "../../types/log";
import { CDPSession } from "playwright";
export declare function formatSimplifiedTree(node: AccessibilityNode, level?: number): string;
/**
 * Builds a hierarchical tree structure from a flat array of accessibility nodes.
 * The function processes nodes in multiple passes to create a clean, meaningful tree.
 * @param nodes - Flat array of accessibility nodes from the CDP
 * @returns Object containing both the tree structure and a simplified string representation
 */
export declare function buildHierarchicalTree(nodes: AccessibilityNode[]): TreeResult;
export declare function getAccessibilityTree(page: StagehandPage, logger: (logLine: LogLine) => void): Promise<TreeResult>;
export declare function getXPathByResolvedObjectId(cdpClient: CDPSession, resolvedObjectId: string): Promise<string>;
